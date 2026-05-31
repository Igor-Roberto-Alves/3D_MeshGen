import torch
import torch.nn as nn
from typing import Tuple, Optional


def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Iterative FPS on a batch of point clouds.
    xyz : (B, N, 3)
    returns indices (B, n_samples)
    """

    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    # start from a random point per batch
    centroid = torch.randint(0, N, (B,), device=device)
    for i in range(n_samples):
        idx[:, i] = centroid
        c = xyz[torch.arange(B, device=device), centroid].unsqueeze(1)  # (B,1,3)
        d = torch.sum((xyz - c) ** 2, dim=-1)  # (B,N)
        dist = torch.min(dist, d)
        centroid = dist.argmax(dim=-1)
    return idx


def knn_group(
    xyz: torch.Tensor, centers: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each center point find its k nearest neighbours in xyz.
    xyz     : (B, N, 3)
    centers : (B, M, 3)
    returns:
        grouped_xyz  (B, M, k, 3)  – localised relative coords
        grouped_idx  (B, M, k)
    """
    B, N, _ = xyz.shape
    M = centers.shape[1]
    # pairwise squared distances: (B, M, N)
    diff = centers.unsqueeze(2) - xyz.unsqueeze(1)  # (B,M,N,3)
    sq_dist = (diff**2).sum(-1)  # (B,M,N)
    idx = sq_dist.topk(k, largest=False, dim=-1).indices  # (B,M,k)
    # gather neighbours
    idx_flat = idx.reshape(B, -1)  # (B, M*k)
    pts = torch.gather(xyz, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3))
    grouped = pts.reshape(B, M, k, 3)
    grouped -= centers.unsqueeze(2)  # relative coords
    return grouped, idx


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOKEN EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """
    Mini-PointNet per group  →  one token per group centre.
    Input:  (B, M, k, 3)
    Output: (B, M, C)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 256, k: int = 32):
        super().__init__()
        mid = out_ch // 2
        self.net = nn.Sequential(
            nn.Linear(in_ch, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Linear(mid, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.k = k
        self.out_ch = out_ch

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        # grouped: (B, M, k, 3)
        B, M, k, C = grouped.shape
        x = grouped.reshape(B * M * k, C)
        x = self.net(x).reshape(B * M, k, self.out_ch)
        x = x.max(dim=1).values  # max-pool over k pts
        return x.reshape(B, M, self.out_ch)


class PositionalEncoding(nn.Module):
    """Lightweight MLP positional encoding from 3-D coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRANSFORMER BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────


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
        kv = x if kv is None else kv
        attn_out, _ = self.attn(self.norm1(x), self.norm1(kv), self.norm1(kv))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HIERARCHICAL ENCODER  (blue branch)
# ─────────────────────────────────────────────────────────────────────────────


class HierarchicalEncoder(nn.Module):
    """
    Three FPS + kNN grouping levels, each producing tokens that are
    concatenated and projected into d_model.
    """

    LEVELS = [
        dict(n_samples=512, k=32),
        dict(n_samples=256, k=32),
        dict(n_samples=128, k=32),
    ]

    def __init__(self, d_model: int = 384, n_heads: int = 6, depth: int = 4):
        super().__init__()
        self.d_model = d_model

        # One PatchEmbed per level
        self.embeds = nn.ModuleList(
            [PatchEmbed(in_ch=3, out_ch=d_model, k=lv["k"]) for lv in self.LEVELS]
        )
        # Positional encodings
        self.pos_encs = nn.ModuleList(
            [PositionalEncoding(d_model) for _ in self.LEVELS]
        )
        # Shared transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        # Final projection after concat of all levels
        self.proj = nn.Linear(d_model * len(self.LEVELS), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz : (B, N, 3)
        returns:
            tokens  (B, M_total, d_model)
            centers (B, M_total, 3)
        """
        all_tokens, all_centers = [], []
        for i, lv in enumerate(self.LEVELS):
            idx = farthest_point_sampling(xyz, lv["n_samples"])
            centers = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            grouped, _ = knn_group(xyz, centers, lv["k"])
            tokens = self.embeds[i](grouped) + self.pos_encs[i](centers)
            all_tokens.append(tokens)
            all_centers.append(centers)

        # Run transformer per level, then mean-pool → aggregate
        encoded_levels = []
        for i in range(len(self.LEVELS)):
            t = all_tokens[i]
            for blk in self.blocks:
                t = blk(t)
            encoded_levels.append(t.mean(dim=1))  # (B, d_model)

        centers = torch.cat(all_centers, dim=1)  # (B, M_total, 3)
        feat = torch.cat(encoded_levels, dim=-1)  # (B, d_model * 3)
        feat = self.norm(self.proj(feat))  # (B, d_model)
        return feat, centers


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GLOBAL ENCODER  (green branch)
# ─────────────────────────────────────────────────────────────────────────────


class GlobalEncoder(nn.Module):
    """
    Single-level FPS grouping + transformer.  Captures global shape context.
    """

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
        self.embed = PatchEmbed(in_ch=3, out_ch=d_model, k=k)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz : (B, N, 3)  →  (B, d_model)"""
        idx = farthest_point_sampling(xyz, self.n_tokens)
        centers = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        grouped, _ = knn_group(xyz, centers, k=32)
        tokens = self.embed(grouped) + self.pos_enc(centers)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-BRANCH ATTENTION FUSION
# ─────────────────────────────────────────────────────────────────────────────


class CrossBranchFusion(nn.Module):
    """
    Bidirectional cross-attention between hierarchical and global feature vectors.
    Both are unsqueezed to sequence length 1 for nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attn_h2g = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_g2h = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """h, g : (B, d_model)  →  fused (B, d_model)"""
        h_s = h.unsqueeze(1)  # (B, 1, d_model)
        g_s = g.unsqueeze(1)
        h2g, _ = self.attn_h2g(self.norm_h(h_s), self.norm_g(g_s), self.norm_g(g_s))
        g2h, _ = self.attn_g2h(self.norm_g(g_s), self.norm_h(h_s), self.norm_h(h_s))
        h_out = (h_s + h2g).squeeze(1)
        g_out = (g_s + g2h).squeeze(1)
        return self.proj(torch.cat([h_out, g_out], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VAE BOTTLENECK
# ─────────────────────────────────────────────────────────────────────────────


class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # Clamp logvar for training stability
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparametrisation
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(μ,σ²) ‖ N(0,1) )  averaged over the batch."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GENERATIVE DECODER
# ─────────────────────────────────────────────────────────────────────────────


class PointDecoder(nn.Module):
    """
    Transformer-based point cloud generator.
    Learned query tokens attend to the latent code z to produce N × 3 points.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 4,
        n_out: int = 2048,
    ):
        super().__init__()
        self.n_out = n_out
        # Project z to d_model memory
        self.z_proj = nn.Linear(latent_dim, d_model)
        # Learned query tokens (one per output point — grouped by centres)
        self.n_centers = n_out // 32  # e.g. 64 centres × 32 pts/centre
        self.query = nn.Parameter(torch.randn(1, self.n_centers, d_model) * 0.02)
        # Cross-attention layers: queries attend to z
        self.cross_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        # Self-attention refinement
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)
        # Per-token MLP to predict 32 × 3 point offsets
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 32 * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, latent_dim)  →  (B, n_out, 3)"""
        B = z.shape[0]
        mem = self.z_proj(z).unsqueeze(1)  # (B, 1, d_model)
        q = self.query.expand(B, -1, -1)  # (B, n_centers, d_model)
        for blk in self.cross_blocks:
            q = blk(q, kv=mem)
        for blk in self.self_blocks:
            q = blk(q)
        q = self.norm(q)  # (B, n_centers, d_model)
        pts = self.head(q)  # (B, n_centers, 32*3)
        pts = pts.reshape(B, self.n_centers, 32, 3)
        # Normalise to unit sphere
        pts = pts - pts.mean(dim=(1, 2), keepdim=True)
        pts = pts / (pts.norm(dim=-1, keepdim=True).max() + 1e-8)
        return pts.reshape(B, self.n_out, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────


class DualBranchPointVAE(nn.Module):
    """
    Full dual-branch VAE.

    Args:
        d_model      : transformer channel dimension
        latent_dim   : size of the VAE latent space
        n_out        : number of output points (reconstruction)
        enc_depth    : transformer layers in each encoder branch
        dec_depth    : transformer layers in decoder
        beta         : β-VAE weight on the KL term
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 512,
        n_out: int = 2048,
        enc_depth: int = 4,
        dec_depth: int = 4,
        n_heads: int = 6,
        beta: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim

        # Shared-weight Siamese encoders
        self.hier_enc = HierarchicalEncoder(
            d_model=d_model, n_heads=n_heads, depth=enc_depth
        )
        self.glob_enc = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion = CrossBranchFusion(d_model=d_model, n_heads=n_heads)
        self.bottleneck = VAEBottleneck(in_dim=d_model, latent_dim=latent_dim)
        self.decoder = PointDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=dec_depth,
            n_out=n_out,
        )

    # ── forward ──────────────────────────────────────────────────────────────
    def encode(self, xyz: torch.Tensor):
        h_feat, _ = self.hier_enc(xyz)
        g_feat = self.glob_enc(xyz)
        fused = self.fusion(h_feat, g_feat)
        z, mu, logvar = self.bottleneck(fused)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, xyz: torch.Tensor) -> dict:
        z, mu, logvar = self.encode(xyz)
        recon = self.decode(z)
        return dict(recon=recon, z=z, mu=mu, logvar=logvar)

    # ── loss ─────────────────────────────────────────────────────────────────
    def loss(self, out: dict, target: torch.Tensor) -> dict:
        cd = chamfer_distance(out["recon"], target)
        kl = VAEBottleneck.kl_loss(out["mu"], out["logvar"])
        total = cd + self.beta * kl
        return dict(total=total, cd=cd, kl=kl)

    # ── generation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n point clouds from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self, xyz_a: torch.Tensor, xyz_b: torch.Tensor, steps: int = 8
    ) -> torch.Tensor:
        """Spherical linear interpolation between two shapes."""
        _, mu_a, _ = self.encode(xyz_a)
        _, mu_b, _ = self.encode(xyz_b)
        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            z = (1 - a) * mu_a + a * mu_b
            shapes.append(self.decode(z))
        return torch.stack(shapes, dim=1)  # (B, steps, N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CHAMFER DISTANCE  (bidirectional, L2)
# ─────────────────────────────────────────────────────────────────────────────


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Chamfer Distance (L2 version).
    pred, gt : (B, N, 3)
    """
    # (B, N_pred, N_gt)
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist = (diff**2).sum(-1)  # (B, N_pred, N_gt)
    # pred → gt: for each pred pt, nearest gt pt
    d1 = dist.min(dim=2).values.mean(dim=1)  # (B,)
    # gt → pred
    d2 = dist.min(dim=1).values.mean(dim=1)  # (B,)
    return (d1 + d2).mean()
