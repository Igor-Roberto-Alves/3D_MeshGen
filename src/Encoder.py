from torch import nn
import torch
from typing import Tuple, Optional
from src.Tokenizer import (
    PatchEmbed,
    PositionalEncoding,
    farthest_point_sampling,
    knn_group,
)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kv_src = x if kv is None else kv
        attn_out, _ = self.attn(self.norm1(x), self.norm1(kv_src), self.norm1(kv_src))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── Blue Branch ───────────────────────────────────────────────────────────────


class HierarchicalEncoder(nn.Module):
    """
    Three FPS + kNN grouping levels whose tokens are max-pooled per level,
    projected together, and returned as a single global descriptor (B, d_model).
    """

    LEVELS = [
        dict(n_samples=512, k=32),
        dict(n_samples=256, k=32),
        dict(n_samples=128, k=32),
    ]

    def __init__(self, d_model: int = 384, n_heads: int = 6, depth: int = 4):
        super().__init__()
        self.d_model = d_model

        self.embeds = nn.ModuleList(
            [PatchEmbed(in_ch=6, out_ch=d_model, k=lv["k"]) for lv in self.LEVELS]
        )
        self.pos_encs = nn.ModuleList(
            [PositionalEncoding(d_model) for _ in self.LEVELS]
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        # Projects the concatenation of the 3 per-level max-pooled descriptors
        self.proj = nn.Linear(d_model * len(self.LEVELS), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz : (B, N, 6)
        Returns
        -------
        feat_global : (B, d_model)
        centers     : (B, M_total, 3)
        """

        all_tokens, all_centers_xyz = [], []

        for i, lv in enumerate(self.LEVELS):
            idx = farthest_point_sampling(xyz, lv["n_samples"])
            centers = torch.gather(
                xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.shape[-1])
            )
            grouped, _ = knn_group(xyz, centers, lv["k"])
            tokens = self.embeds[i](grouped) + self.pos_encs[i](centers[..., :3])
            all_tokens.append(tokens)
            all_centers_xyz.append(centers[..., :3])

        # Encode each level independently through the shared transformer blocks
        encoded_levels = []
        for i in range(len(self.LEVELS)):
            t = all_tokens[i]
            for blk in self.blocks:
                t = blk(t)
            encoded_levels.append(t)  # (B, M_level, d_model)

        # Max-pool each level → (B, d_model), concat → proj → norm  (Bug 6 fix)
        level_globals = [t.max(dim=1).values for t in encoded_levels]
        feat_global = self.norm(self.proj(torch.cat(level_globals, dim=-1)))

        centers = torch.cat(all_centers_xyz, dim=1)  # (B, M_total, 3)
        return feat_global, centers


# ── Green Branch ─────────────────────────────────────────────────────────────


class GlobalEncoder(nn.Module):
    """Single-level FPS grouping + transformer. Captures global shape context."""

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 4,
        n_tokens: int = 64,
        k: int = 32,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.k = k
        self.embed = PatchEmbed(in_ch=6, out_ch=d_model, k=k)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz : (B, N, 6)  →  (B, d_model)"""
        idx = farthest_point_sampling(xyz, self.n_tokens)
        centers = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.shape[-1]))
        grouped, _ = knn_group(xyz, centers, k=self.k)
        tokens = self.embed(grouped) + self.pos_enc(centers[..., :3])
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens.max(dim=1).values)  # (B, d_model)


# ── Cross-attention used in CrossBranchFusion ─────────────────────────────────


class Attention(nn.Module):
    """
    Scaled dot-product cross-attention.
    x  → query
    y  → key / value  (defaults to x for self-attention)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is None:
            y = x

        B, Ny, C = y.shape
        kv = (
            self.kv(y)
            .reshape(B, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        B, Nx, C = x.shape
        q = (
            self.q(x)
            .reshape(B, Nx, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(1) * -1e5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── Cross-Branch Fusion ───────────────────────────────────────────────────────


class CrossBranchFusion(nn.Module):
    """
    Fuses hierarchical (h) and global (g) descriptors in two stages:

    Stage 1 — Cross-attention
        h is the query: hierarchical features decide what to pull from g.
        g is the key/value: provides global context.
        A residual on h preserves the hierarchical signal.

    Stage 2 — MLP residual
        The attention output and g are concatenated and mixed by a small MLP.
        A final residual on g ensures the global signal is never discarded.

    Both stages use Pre-LN (LayerNorm before each sub-layer).
    """

    def __init__(self, d_model: int, n_heads: int = 8, **kwargs):
        super().__init__()
        # Pre-LN for the cross-attention inputs
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads=n_heads)
        self.norm_mid = nn.LayerNorm(d_model)  # after attn residual

        # MLP stage
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        h : (B, d_model)  hierarchical descriptor
        g : (B, d_model)  global descriptor
        Returns fused (B, d_model)
        """
        # Attention operates on sequences; lift (B, C) → (B, 1, C)
        h_seq = self.norm_h(h).unsqueeze(1)  # (B, 1, d_model)  query
        g_seq = self.norm_g(g).unsqueeze(1)  # (B, 1, d_model)  key / value

        # Stage 1: h attends to g, residual on h
        attended = self.attn(h_seq, y=g_seq).squeeze(1)  # (B, d_model)
        attended = self.norm_mid(attended + h)  # (B, d_model)

        # Stage 2: MLP mix, residual on g
        fused = self.fusion_mlp(torch.cat([attended, g], dim=-1))  # (B, d_model)
        return self.norm_out(fused + g)


# ── VAE Bottleneck ────────────────────────────────────────────────────────────


class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparametrisation
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(μ,σ²) ‖ N(0,1) ) averaged over the batch."""

        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
