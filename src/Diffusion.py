import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class CosineSchedule(nn.Module):


    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__()
        t   = torch.arange(T + 1, dtype=torch.float64) # Array 0 até T
        f   = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        acp = (f / f[0]).clamp(1e-5, 1 - 1e-5).float()   # ᾱ, T+1 values

        betas  = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        acp    = acp[1:]   # drop t=0, now T values

        self.T = T
        self.register_buffer('betas',          betas)
        self.register_buffer('alphas',         alphas)
        self.register_buffer('acp',            acp)
        self.register_buffer('sqrt_acp',       acp.sqrt())
        self.register_buffer('sqrt_one_m_acp', (1 - acp).sqrt())

    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        """Forward process: xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_acp[t]
        s2 = self.sqrt_one_m_acp[t]
        for _ in range(x0.dim() - 1):
            s1 = s1.unsqueeze(-1)
            s2 = s2.unsqueeze(-1)
        return s1 * x0 + s2 * noise, noise

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        denoiser:  nn.Module,
        shape:     tuple,          # shape WITHOUT batch dim, e.g. (256,) or (2048, 6)
        condition: torch.Tensor,   # (B, ...)
        uncond:    torch.Tensor | None = None,
        guidance:  float = 1.0,
        device:    torch.device | None = None,
    ) -> torch.Tensor:
        B = condition.shape[0]
        if device is None:
            device = condition.device
        x = torch.randn(B, *shape, device=device)

        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self._p_step(denoiser, x, t, condition, uncond, guidance)
        return x

    def _p_step(self, denoiser, xt, t, cond, uncond, guidance):
        eps = denoiser(xt, t, cond)
        if guidance != 1.0 and uncond is not None:
            eps_u = denoiser(xt, t, uncond)
            eps   = eps_u + guidance * (eps - eps_u)

        # Compute all (B,) scalars BEFORE broadcasting to match xt dims.
        beta_t   = self.betas[t]
        alpha_t  = self.alphas[t]
        acp_t    = self.acp[t]
        acp_prev = torch.where(t > 0, self.acp[t - 1], torch.ones_like(acp_t))

        def bcast(v):
            for _ in range(xt.dim() - 1):
                v = v.unsqueeze(-1)
            return v

        beta_t, alpha_t, acp_t, acp_prev = map(bcast, [beta_t, alpha_t, acp_t, acp_prev])

        x0_pred = (xt - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()
        x0_pred = x0_pred.clamp(-10, 10)

        mean = (beta_t * acp_prev.sqrt() / (1 - acp_t)) * x0_pred \
             + ((1 - acp_prev) * alpha_t.sqrt() / (1 - acp_t)) * xt

        var  = beta_t * (1 - acp_prev) / (1 - acp_t)
        mask = bcast((t > 0).float())
        return mean + mask * var.clamp(min=1e-20).sqrt() * torch.randn_like(xt)



def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)


class MLPResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * 4)
        self.fc2  = nn.Linear(dim * 4, dim)
        self.cond = nn.Linear(cond_dim, dim * 2)   # → scale + shift
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h  = self.norm(x)
        sc, sh = self.cond(cond).chunk(2, dim=-1)
        h  = h * (1 + sc) + sh
        return x + self.fc2(F.silu(self.fc1(h)))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):

    def __init__(self, dim: int, cond_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = \
            self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, point_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada  = nn.Linear(cond_dim, 2 * dim)
        self.proj = nn.Linear(dim, point_dim)
        nn.init.zeros_(self.ada.weight);  nn.init.zeros_(self.ada.bias)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))



class StyleDenoiser(nn.Module):

    def __init__(
        self,
        style_dim:   int   = 256,
        num_classes: int   = 55,
        hidden:      int   = 512,
        n_layers:    int   = 6,
        T:           int   = 1000,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self.uncond_idx  = num_classes 
        time_dim         = hidden

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)  
        self.cond_proj = nn.Linear(time_dim * 2, time_dim)

        self.input_proj  = nn.Linear(style_dim, hidden)
        self.blocks      = nn.ModuleList([MLPResBlock(hidden, time_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, style_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _encode_cond(self, t: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_emb(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_idx)
        return self.cond_proj(torch.cat([t_emb, c_emb], dim=-1))

    def forward(self, zt: torch.Tensor, t: torch.Tensor,
                class_idx: torch.Tensor) -> torch.Tensor:
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(class_idx.shape, device=class_idx.device) < self.cfg_dropout
            class_idx = class_idx.masked_fill(drop, self.uncond_idx)

        cond = self._encode_cond(t, class_idx)
        h    = self.input_proj(zt)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)

    def uncond(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.full((B,), self.uncond_idx, device=device, dtype=torch.long)


class LatentPointDenoiser(nn.Module):
 

    def __init__(
        self,
        point_dim:  int   = 8,     # must match Vae latent_dim — no anchors in z_l
        style_dim:  int   = 256,
        hidden:     int   = 256,
        n_layers:   int   = 8,
        n_heads:    int   = 8,      # attention heads per DiT block (must divide hidden)
        mlp_ratio:  float = 4.0,    # FFN expansion inside each DiT block
        T:          int   = 1000,
    ):
        super().__init__()
        assert hidden % n_heads == 0, (
            f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
        )
        time_dim = hidden

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        # Fuse time embedding + global style into a single conditioning vector.
        self.cond_proj = nn.Linear(time_dim + style_dim, time_dim)

        self.input_proj = nn.Linear(point_dim, hidden)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, time_dim, n_heads, mlp_ratio) for _ in range(n_layers)
        ])
        self.final = DiTFinalLayer(hidden, time_dim, point_dim)

    def forward(self, zt: torch.Tensor, t: torch.Tensor,
                z_g: torch.Tensor) -> torch.Tensor:
        # zt:  (B, N, point_dim)
        # t:   (B,)  long
        # z_g: (B, style_dim)
        t_emb = sinusoidal_emb(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        cond  = self.cond_proj(torch.cat([t_emb, z_g], dim=-1))   # (B, time_dim)

        h = self.input_proj(zt)          # (B, N, hidden)
        for block in self.blocks:
            h = block(h, cond)
        return self.final(h, cond)       # (B, N, point_dim)