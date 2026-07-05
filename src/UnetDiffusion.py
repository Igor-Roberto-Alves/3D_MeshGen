"""
UnetDiffusion.py
----------------
Conditional 1-D U-Net denoiser for DDPM over FLAT latent vectors
(the [B, latent_dim] mu produced by VaePointnet / EncoderPointnet,
latent_dim = 256 or 512 depending on the VAE configuration).

The latent vector is treated as a 1-channel 1-D signal [B, 1, D] and
processed by a standard DDPM U-Net (ResBlocks + self-attention +
down/upsampling with skip connections). Conditioning:

  timestep  — sinusoidal embedding -> MLP
  class     — nn.Embedding (2 classes: 0 = airplane, 1 = car, plus a
              learned "null" class for classifier-free guidance)

Both embeddings are summed and injected into every ResBlock via
FiLM (scale + shift on the GroupNorm output).

Interface is identical to StyleDenoiser in src/Diffusion.py:

    eps = model(z_t, t, class_idx)          # z_t: [B, D], t: [B], idx: [B]
    null = model.uncond(B, device)          # null-class indices for CFG

so it plugs directly into CosineSchedule.sample() (full DDPM chain) or
into ddim_sample() below (fast deterministic sampling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Diffusion import CosineSchedule, sinusoidal_emb  # noqa: F401 (re-exported)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """GroupNorm → SiLU → Conv1d twice, FiLM-conditioned, with residual skip."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)      # → scale + shift

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttnBlock1D(nn.Module):
    """Self-attention over the positions of the 1-D signal (pre-norm, residual)."""

    def __init__(self, ch: int, n_heads: int = 4, groups: int = 8):
        super().__init__()
        assert ch % n_heads == 0, f"ch ({ch}) must be divisible by n_heads ({n_heads})"
        self.norm = nn.GroupNorm(groups, ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.proj = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)                    # [B, L, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.proj(h.transpose(1, 2))


class _Block(nn.Module):
    """ResBlock followed by optional attention — one U-Net step."""

    def __init__(self, in_ch, out_ch, emb_dim, use_attn, n_heads, groups):
        super().__init__()
        self.res  = ResBlock1D(in_ch, out_ch, emb_dim, groups)
        self.attn = AttnBlock1D(out_ch, n_heads, groups) if use_attn else None

    def forward(self, x, emb):
        x = self.res(x, emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ─────────────────────────────────────────────────────────────────────────────
# Conditional 1-D U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    eps-prediction U-Net over flat latent vectors.

    z_t  [B, latent_dim]  +  t [B]  +  class_idx [B]   →   eps [B, latent_dim]

    Classifier-free guidance: during training each class label is replaced
    by the null index with prob. `cfg_dropout`; at sampling time pass
    `uncond=model.uncond(B, device)` and guidance > 1 to CosineSchedule.sample
    or ddim_sample.
    """

    def __init__(
        self,
        latent_dim:   int   = 256,
        num_classes:  int   = 2,
        base_ch:      int   = 64,
        ch_mult:      tuple = (1, 2, 4),
        n_res_blocks: int   = 2,
        attn_levels:  tuple = (1, 2),   # U-Net levels that get self-attention
        n_heads:      int   = 4,
        time_dim:     int   = 128,
        cfg_dropout:  float = 0.1,
        groups:       int   = 8,
    ):
        super().__init__()
        n_levels = len(ch_mult)
        assert latent_dim % (2 ** (n_levels - 1)) == 0, (
            f"latent_dim ({latent_dim}) must be divisible by 2^{n_levels - 1} "
            f"for {n_levels} U-Net levels"
        )

        self.latent_dim  = latent_dim
        self.cfg_dropout = cfg_dropout
        self.uncond_idx  = num_classes           # extra "null" class for CFG
        self.time_dim    = time_dim

        emb_dim = time_dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        chs = [base_ch * m for m in ch_mult]

        # ── Encoder (down path) ───────────────────────────────────────────
        self.stem = nn.Conv1d(1, base_ch, 3, padding=1)

        self.downs       = nn.ModuleList()
        self.downsamples = nn.ModuleList()   # len == n_levels - 1
        skip_chs = [base_ch]
        ch = base_ch
        for i, out_ch in enumerate(chs):
            stage = nn.ModuleList()
            for _ in range(n_res_blocks):
                stage.append(_Block(ch, out_ch, emb_dim, i in attn_levels, n_heads, groups))
                ch = out_ch
                skip_chs.append(ch)
            self.downs.append(stage)
            if i < n_levels - 1:
                self.downsamples.append(Downsample1D(ch))
                skip_chs.append(ch)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.mid1     = ResBlock1D(ch, ch, emb_dim, groups)
        self.mid_attn = AttnBlock1D(ch, n_heads, groups)
        self.mid2     = ResBlock1D(ch, ch, emb_dim, groups)

        # ── Decoder (up path, skip-concat) ────────────────────────────────
        self.ups       = nn.ModuleList()
        self.upsamples = nn.ModuleList()     # len == n_levels - 1
        for i, out_ch in reversed(list(enumerate(chs))):
            stage = nn.ModuleList()
            for _ in range(n_res_blocks + 1):
                skip_ch = skip_chs.pop()
                stage.append(_Block(ch + skip_ch, out_ch, emb_dim,
                                    i in attn_levels, n_heads, groups))
                ch = out_ch
            self.ups.append(stage)
            if i > 0:
                self.upsamples.append(Upsample1D(ch))

        # ── Output head (zero-init → identity at start of training) ──────
        self.out_norm = nn.GroupNorm(groups, base_ch)
        self.out_conv = nn.Conv1d(base_ch, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------
    def _encode_cond(self, t: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_emb(t, self.time_dim)
        return self.time_mlp(t_emb) + self.class_emb(class_idx)

    # ------------------------------------------------------------------
    def forward(self, zt: torch.Tensor, t: torch.Tensor,
                class_idx: torch.Tensor) -> torch.Tensor:
        # CFG label dropout — only while training
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(class_idx.shape, device=class_idx.device) < self.cfg_dropout
            class_idx = class_idx.masked_fill(drop, self.uncond_idx)

        emb = self._encode_cond(t, class_idx)               # [B, emb_dim]

        h  = self.stem(zt.unsqueeze(1))                     # [B, base_ch, D]
        hs = [h]

        for i, stage in enumerate(self.downs):
            for block in stage:
                h = block(h, emb)
                hs.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
                hs.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        for j, stage in enumerate(self.ups):
            for block in stage:
                h = block(torch.cat([h, hs.pop()], dim=1), emb)
            if j < len(self.upsamples):
                h = self.upsamples[j](h)

        out = self.out_conv(F.silu(self.out_norm(h)))       # [B, 1, D]
        return out.squeeze(1)

    # ------------------------------------------------------------------
    def uncond(self, B: int, device: torch.device) -> torch.Tensor:
        """Null-class indices for classifier-free guidance."""
        return torch.full((B,), self.uncond_idx, device=device, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# DDIM sampling (fast alternative to CosineSchedule.sample's full DDPM chain)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    schedule:  CosineSchedule,
    denoiser:  nn.Module,
    shape:     tuple,                       # WITHOUT batch dim, e.g. (256,)
    condition: torch.Tensor,                # [B] class indices
    uncond:    torch.Tensor | None = None,  # [B] null indices (CFG)
    guidance:  float = 1.0,
    steps:     int   = 100,
    eta:       float = 0.0,                 # 0 = deterministic DDIM
    device:    torch.device | None = None,
) -> torch.Tensor:
    B = condition.shape[0]
    if device is None:
        device = condition.device

    ts = torch.linspace(0, schedule.T - 1, steps, device=device).long().flip(0)
    x  = torch.randn(B, *shape, device=device)

    for i in range(len(ts)):
        t      = ts[i]
        t_prev = ts[i + 1] if i + 1 < len(ts) else None
        t_b    = torch.full((B,), int(t), device=device, dtype=torch.long)

        eps = denoiser(x, t_b, condition)
        if guidance != 1.0 and uncond is not None:
            eps_u = denoiser(x, t_b, uncond)
            eps   = eps_u + guidance * (eps - eps_u)

        acp_t    = schedule.acp[t]
        acp_prev = schedule.acp[t_prev] if t_prev is not None else torch.ones_like(acp_t)

        x0 = (x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()
        x0 = x0.clamp(-10, 10)

        sigma = eta * ((1 - acp_prev) / (1 - acp_t)).sqrt() \
                    * (1 - acp_t / acp_prev).clamp(min=0).sqrt()
        dir_x = (1 - acp_prev - sigma ** 2).clamp(min=0).sqrt() * eps

        x = acp_prev.sqrt() * x0 + dir_x
        if eta > 0 and t_prev is not None:
            x = x + sigma * torch.randn_like(x)

    return x
