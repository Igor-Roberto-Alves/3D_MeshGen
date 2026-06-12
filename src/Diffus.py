import torch
import torch.nn.functional as F
import math


class DDPMScheduler:


    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.T = T

        if schedule == "cosine":
            betas = self._cosine_betas(T)
        else:
            betas = torch.linspace(beta_start, beta_end, T)

        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register("betas",          betas)
        self.register("alphas",         alphas)
        self.register("alpha_bar",      alpha_bar)
        self.register("alpha_bar_prev", alpha_bar_prev)
        self.register("sqrt_alpha_bar",       alpha_bar.sqrt())
        self.register("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register("posterior_variance",
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    @staticmethod
    def _cosine_betas(T: int, s: float = 8e-3) -> torch.Tensor:
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        f     = f / f[0]
        betas = 1.0 - f[1:] / f[:-1]
        return betas.clamp(0, 0.999).float()


    def q_sample(
        self,
        x0:    torch.Tensor,  # (..., D)
        t:     torch.Tensor,  # (B,)  long
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_t, noise)."""
        if noise is None:
            noise = torch.randn_like(x0)

        shape = (-1,) + (1,) * (x0.ndim - 1)          # broadcast over D / N,D
        sqrt_ab  = self.sqrt_alpha_bar[t].to(x0).view(shape)
        sqrt_1ab = self.sqrt_one_minus_alpha_bar[t].to(x0).view(shape)
        return sqrt_ab * x0 + sqrt_1ab * noise, noise


    @torch.no_grad()
    def p_sample(
        self,
        model_output: torch.Tensor,   # predicted noise ε̂  (..., D)
        x_t:          torch.Tensor,   # (..., D)
        t:            torch.Tensor,   # (B,)  long
    ) -> torch.Tensor:
        shape = (-1,) + (1,) * (x_t.ndim - 1)

        beta          = self.betas[t].to(x_t).view(shape)
        alpha         = self.alphas[t].to(x_t).view(shape)
        sqrt_1ab      = self.sqrt_one_minus_alpha_bar[t].to(x_t).view(shape)
        post_var      = self.posterior_variance[t].to(x_t).view(shape)

        mean  = (1.0 / alpha.sqrt()) * (x_t - beta / sqrt_1ab * model_output)
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        return mean + post_var.sqrt() * noise

    # ------------------------------------------------------------------ #
    # Full denoising loop (inference)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        model,                        # callable: (x_t, t, **cond) -> ε̂
        shape: tuple,
        device: torch.device,
        **cond_kwargs,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps_hat = model(x, t, **cond_kwargs)
            x = self.p_sample(eps_hat, x, t)
        return x

    # ------------------------------------------------------------------ #
    # Training loss: simple MSE on noise prediction
    # ------------------------------------------------------------------ #
    def training_loss(
        self,
        model,                         # callable: (x_t, t, **cond) -> ε̂
        x0:    torch.Tensor,
        **cond_kwargs,
    ) -> torch.Tensor:
        B      = x0.shape[0]
        t      = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        eps_hat    = model(x_t, t, **cond_kwargs)
        return F.mse_loss(eps_hat, noise)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SinusoidalEmbedding(nn.Module):
    """Timestep → continuous sinusoidal embedding → linear projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,) long → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)
        return self.proj(emb)


class ResBlock(nn.Module):
    """Residual MLP block with optional timestep conditioning."""

    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.t_proj = nn.Linear(t_dim, dim)
        self.norm2  = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x) + self.t_proj(t_emb)
        h = self.norm2(h)
        return x + self.ff(h)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention on a (B, D) vector treated as (B, 1, D)."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D)
        x2 = x.unsqueeze(1)                              # (B, 1, D)
        out, _ = self.attn(x2, x2, x2)
        return self.norm(x + out.squeeze(1))


class CrossAttention1D(nn.Module):
    """
    x (query)   : (B, D_x)
    ctx (key/val): (B, D_ctx)
    Returns     : (B, D_x)
    """

    def __init__(self, query_dim: int, ctx_dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(query_dim, heads, batch_first=True, kdim=ctx_dim, vdim=ctx_dim, dropout=0.0)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q   = x.unsqueeze(1)    # (B, 1, D_x)
        kv  = ctx.unsqueeze(1)  # (B, 1, D_ctx)
        out, _ = self.attn(q, kv, kv)
        return self.norm(x + out.squeeze(1))


# ======================================================================
# StyleScoreNet — prior for p(z_style)
# ======================================================================

class StyleScoreNet(nn.Module):
    """
    Predicts the noise added to z_style.

    Parameters
    ----------
    style_dim : dimension of the style latent vector
    hidden_dim: inner width of the MLP blocks
    depth     : number of ResBlocks
    heads     : attention heads
    """

    def __init__(
        self,
        style_dim:  int = 512,
        hidden_dim: int = 512,
        depth:      int = 6,
        heads:      int = 4,
        num_classes: int = 53
    ):
        super().__init__()
        self.t_emb     = SinusoidalEmbedding(hidden_dim)
        self.cls_emb   = nn.Embedding(num_classes, hidden_dim)  # Embedding das classes
        self.in_proj   = nn.Linear(style_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        # Attention every 2 blocks
        self.attns  = nn.ModuleList([
            SelfAttention1D(hidden_dim, heads) if i % 2 == 1 else nn.Identity()
            for i in range(depth)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, style_dim)
        # Zero-init output projection (improves early training stability)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,cls: torch.Tensor) -> torch.Tensor:   # ← cls (B,) long
        t_emb   = self.t_emb(t)
        cls_emb = self.cls_emb(cls)      
        h       = self.in_proj(z_t)

        h = h + t_emb + cls_emb            

        for blk, attn in zip(self.blocks, self.attns):
            h = blk(h, t_emb + cls_emb)     
            h = attn(h)

        return self.out_proj(self.out_norm(h))

# ======================================================================
# PointScoreNet — conditional prior for p(z_points | z_style)
# ======================================================================
class PointScoreNet(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        style_dim:  int = 512,
        hidden_dim: int = 512,
        depth:      int = 8,
        heads:      int = 4,
        # num_classes removido
    ):
        super().__init__()
        self.t_emb      = SinusoidalEmbedding(hidden_dim)
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.in_proj    = nn.Linear(latent_dim, hidden_dim)
        # cls_emb removido

        self.blocks  = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.x_attns = nn.ModuleList([
            CrossAttention1D(hidden_dim, hidden_dim, heads) if i % 2 == 1
            else nn.Identity()
            for i in range(depth)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, t, z_style):   # cls removido
        t_emb = self.t_emb(t)
        ctx   = self.style_proj(z_style)
        h     = self.in_proj(z_t) + t_emb + ctx

        for blk, x_attn in zip(self.blocks, self.x_attns):
            h = blk(h, t_emb)             # cls_emb removido daqui
            if isinstance(x_attn, CrossAttention1D):
                h = x_attn(h, ctx)

        return self.out_proj(self.out_norm(h))


class Model(nn.Module):
    # ... __init__ e encode inalterados ...

    def point_loss(self, z_points, z_style):   # cls removido
        B = z_points.shape[0]
        t = torch.randint(0, self.point_sched.T, (B,), device=z_points.device)
        z_t, noise = self.point_sched.q_sample(z_points, t)
        eps_hat    = self.point_net(z_t, t, z_style=z_style)
        return F.mse_loss(eps_hat, noise)

    def training_step(self, x, cls, w_style=1.0, w_points=1.0):
        z_style, z_points = self.encode(x)
        l_style  = self.style_loss(z_style, cls)
        l_points = self.point_loss(z_points, z_style)   # cls removido
        total    = w_style * l_style + w_points * l_points
        return dict(loss=total, loss_style=l_style, loss_points=l_points)

    @torch.no_grad()
    def sample(self, n, cls: torch.Tensor, device, style_dim=512, latent_dim=256):

        z_style = self.style_sched.sample(
            model  = self.style_net,
            shape  = (n, style_dim),
            device = device,
            cls    = cls,
        )

        def point_model(x_t, t, z_style):          # cls removido
            return self.point_net(x_t, t, z_style=z_style)

        z_points = self.point_sched.sample(
            model   = point_model,
            shape   = (n, latent_dim),
            device  = device,
            z_style = z_style,                      # cls removido
        )

        return self.vae.Decode(z_style, z_points)