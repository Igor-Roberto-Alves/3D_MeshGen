import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import (
    farthest_point_sample, index_points, ball_query,
    AdaGN, SqueezeExcitation, PointSelfAttention,
)


# ---------------------------------------------------------------------------
# Shared voxelisation helper
# ---------------------------------------------------------------------------

def _voxelise(coords: torch.Tensor, features: torch.Tensor, R: int) -> torch.Tensor:
    """
    coords:   (B, 3, N) float, already in [0, R-1]
    features: (B, C, N) float
    Returns:  (B, C, R, R, R)
    """
    B, C, N = features.shape
    device  = features.device
    idx     = coords.long().clamp(0, R - 1)
    flat    = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]  # (B, N)
    feat32  = features.float()
    voxels  = torch.zeros(B, C, R * R * R, device=device, dtype=torch.float32)
    count   = torch.zeros(B, 1, R * R * R, device=device, dtype=torch.float32)
    voxels.scatter_add_(2, flat.unsqueeze(1).expand_as(feat32), feat32)
    count.scatter_add_(2, flat.unsqueeze(1),
                       torch.ones(B, 1, N, device=device, dtype=torch.float32))
    return (voxels / count.clamp(min=1.0)).view(B, C, R, R, R).to(features.dtype)


def _grid_sample_points(voxel_feats: torch.Tensor, xyz_norm: torch.Tensor,
                         R: int, N: int) -> torch.Tensor:
    """
    voxel_feats: (B, C, R, R, R)
    xyz_norm:    (B, 3, N) in [0, R-1]
    Returns:     (B, N, C)
    """
    sample_grid = (xyz_norm / (R - 1) * 2 - 1).clamp(-1.0, 1.0)
    sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)  # (B,1,1,N,3)
    sample_grid = sample_grid.expand(-1, 1, 1, N, 3)
    sampled = F.grid_sample(voxel_feats, sample_grid,
                            mode='bilinear', align_corners=True, padding_mode='border')
    return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # (B, N, C)


# ---------------------------------------------------------------------------
# PVC Block (plain GroupNorm — for GlobalEncoder)
# ---------------------------------------------------------------------------

class PVCBlock(nn.Module):
    """
    Voxel + point branch fused block.
    style_dim=None  → plain GroupNorm (GlobalEncoder).
    style_dim given → AdaGN conditioning (LocalEncoder / Decoder).
    """
    def __init__(self, in_ch: int, out_ch: int, resolution: int,
                 style_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.resolution = resolution
        self.style_dim  = style_dim
        half = out_ch // 2

        # Voxel branch
        mid = max(out_ch, 8)
        self.vox_conv1 = nn.Conv3d(in_ch,  mid,     3, padding=1, bias=False)
        self.vox_gn1   = nn.GroupNorm(min(8, mid), mid)
        self.vox_conv2 = nn.Conv3d(mid,    half,    3, padding=1, bias=False)
        self.vox_gn2   = nn.GroupNorm(min(8, half), half)
        self.vox_se    = SqueezeExcitation(half)

        # Point branch
        self.pt_conv = nn.Conv1d(in_ch, half, 1, bias=False)

        # Normalisation after fusion
        if style_dim is None:
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        else:
            self.norm = AdaGN(out_ch, style_dim, num_groups=min(8, out_ch))

        self.act     = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                style: torch.Tensor | None = None) -> torch.Tensor:
        """
        xyz:   (B, N, 3)   in [-1, 1]
        feat:  (B, N, in_ch)
        style: (B, style_dim) or None
        Returns (B, N, out_ch)
        """
        B, N, _ = xyz.shape
        R       = self.resolution

        coords   = xyz.permute(0, 2, 1)                          # (B, 3, N)
        feats_t  = feat.permute(0, 2, 1)                         # (B, in_ch, N)
        norm_c   = (coords + 1.0) * 0.5 * (R - 1)
        norm_c   = norm_c.clamp(0.0, float(R - 1))

        # Voxel path
        vgrid   = _voxelise(norm_c, feats_t, R)                  # (B, in_ch, R,R,R)
        vx      = self.act(self.vox_gn1(self.vox_conv1(vgrid)))
        vx      = self.vox_gn2(self.vox_conv2(vx))               # (B, half, R,R,R)
        vx_pts  = _grid_sample_points(vx, norm_c, R, N)          # (B, N, half)
        vx_pts  = self.vox_se(vx_pts.permute(0, 2, 1)).permute(0, 2, 1)  # SE on (B,half,N)

        # Point path
        pt      = self.pt_conv(feats_t).permute(0, 2, 1)         # (B, N, half)

        # Fuse
        x = torch.cat([vx_pts, pt], dim=-1).permute(0, 2, 1)    # (B, out_ch, N)

        if self.style_dim is None:
            x = self.act(self.norm(x))
        else:
            x = self.act(self.norm(x, style))

        return self.dropout(x).permute(0, 2, 1)                  # (B, N, out_ch)


# ---------------------------------------------------------------------------
# Set Abstraction
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_centers: int,
                 radius: float, K: int,
                 n_pvc: int, pvc_hidden: int, pvc_res: int,
                 mlp_dims: list[int],
                 use_attention: bool = False, attn_dim: int | None = None,
                 style_dim: int | None = None):
        super().__init__()
        self.n_centers = n_centers
        self.radius    = radius
        self.K         = K

        # PVC blocks operate on all N points BEFORE grouping (Fig 19)
        pvc_layers = []
        ch = in_ch
        for _ in range(n_pvc):
            pvc_layers.append(PVCBlock(ch, pvc_hidden, pvc_res, style_dim=style_dim))
            ch = pvc_hidden
        self.pvc_blocks = nn.ModuleList(pvc_layers)

        # MLP after grouping: input is rel_xyz(3) + grouped_feat(ch)
        mlp_in = 3 + ch
        mlp = []
        for dim in mlp_dims:
            mlp.append(nn.Conv2d(mlp_in, dim, 1, bias=False))
            mlp.append(nn.GroupNorm(min(8, dim), dim))
            mlp.append(nn.SiLU())
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp)
        self.out_ch = mlp_in  # last mlp dim

        self.attn = PointSelfAttention(attn_dim or self.out_ch) if use_attention else None
        # project to attn_dim if needed
        self.attn_proj = (nn.Linear(self.out_ch, attn_dim)
                          if use_attention and attn_dim and attn_dim != self.out_ch
                          else None)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                style: torch.Tensor | None = None):
        """
        xyz:  (B, N, 3)
        feat: (B, N, in_ch)
        Returns: new_xyz (B, M, 3), new_feat (B, M, out_ch)
        """
        B, N, _ = xyz.shape
        M = self.n_centers

        # PVC on full point set
        x = feat
        for pvc in self.pvc_blocks:
            x = pvc(xyz, x, style)   # (B, N, pvc_hidden)

        # FPS
        fps_idx  = farthest_point_sample(xyz, M)          # (B, M)
        new_xyz  = index_points(xyz, fps_idx)              # (B, M, 3)

        # Ball query grouping
        grp_idx  = ball_query(xyz, new_xyz, self.radius, self.K)  # (B, M, K)

        # Gather features + relative coords
        grouped_xyz  = index_points(xyz, grp_idx.view(B, -1)).view(B, M, self.K, 3)
        grouped_feat = index_points(x,   grp_idx.view(B, -1)).view(B, M, self.K, -1)
        rel_xyz      = grouped_xyz - new_xyz.unsqueeze(2)          # (B, M, K, 3)
        grouped_in   = torch.cat([rel_xyz, grouped_feat], dim=-1)  # (B, M, K, 3+C)

        # Conv2d MLP: (B, C, M, K) → max pool over K
        h = grouped_in.permute(0, 3, 1, 2)                        # (B, 3+C, M, K)
        h = self.mlp(h)                                            # (B, out_ch, M, K)
        h = h.max(dim=-1).values.permute(0, 2, 1)                 # (B, M, out_ch)

        if self.attn is not None:
            if self.attn_proj is not None:
                h = self.attn_proj(h)
            h = self.attn(h)

        return new_xyz, h


# ---------------------------------------------------------------------------
# Feature Propagation
# ---------------------------------------------------------------------------

class FeaturePropagation(nn.Module):
    def __init__(self, in_ch_high: int, in_ch_skip: int, out_ch: int,
                 mlp_dims: list[int],
                 n_pvc: int, pvc_hidden: int, pvc_res: int,
                 use_attention: bool = False, attn_dim: int | None = None,
                 style_dim: int | None = None):
        super().__init__()

        mlp_in = in_ch_high + in_ch_skip
        mlp = []
        for dim in mlp_dims:
            mlp.append(nn.Conv1d(mlp_in, dim, 1, bias=False))
            mlp.append(nn.GroupNorm(min(8, dim), dim))
            mlp.append(nn.SiLU())
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp)

        pvc_layers = []
        ch = mlp_in
        for _ in range(n_pvc):
            pvc_layers.append(PVCBlock(ch, pvc_hidden, pvc_res, style_dim=style_dim))
            ch = pvc_hidden
        self.pvc_blocks = nn.ModuleList(pvc_layers)
        self.out_ch = ch

        self.attn = PointSelfAttention(attn_dim or ch) if use_attention else None
        self.attn_proj = (nn.Linear(ch, attn_dim)
                          if use_attention and attn_dim and attn_dim != ch
                          else None)

    def forward(self, xyz_high: torch.Tensor, feat_high: torch.Tensor,
                xyz_low: torch.Tensor,  feat_low: torch.Tensor,
                style: torch.Tensor | None = None):
        """
        Interpolate feat_low (B, M, C_low) defined on xyz_low (B, M, 3)
        to the denser xyz_high (B, N, 3) using inverse-distance weighting (K=3).
        Concatenate with skip feat_high (B, N, C_high), then MLP + PVC.
        """
        B, N, _ = xyz_high.shape
        _, M, _ = xyz_low.shape

        # Inverse-distance weighted interpolation
        dists = torch.cdist(xyz_high.float(), xyz_low.float())    # (B, N, M)
        k     = min(3, M)
        knn_d, knn_i = dists.topk(k, dim=-1, largest=False)       # (B, N, k)

        knn_d  = knn_d.clamp(min=1e-10)
        w      = 1.0 / knn_d                                       # (B, N, k)
        w      = w / w.sum(dim=-1, keepdim=True)                   # normalise

        knn_feat = index_points(feat_low, knn_i.view(B, -1)).view(B, N, k, -1)
        interp   = (knn_feat * w.unsqueeze(-1)).sum(dim=2)         # (B, N, C_low)

        # Concat with skip
        x = torch.cat([interp, feat_high], dim=-1)                 # (B, N, C_low+C_high)
        x = self.mlp(x.permute(0, 2, 1)).permute(0, 2, 1)         # (B, N, mlp_out)

        for pvc in self.pvc_blocks:
            x = pvc(xyz_high, x, style)

        if self.attn is not None:
            if self.attn_proj is not None:
                x = self.attn_proj(x)
            x = self.attn(x)

        return xyz_high, x


# ---------------------------------------------------------------------------
# Global Encoder  (LION Table 5)
# ---------------------------------------------------------------------------

class GlobalEncoder(nn.Module):
    """Encodes the full point cloud into a single global shape latent z_g."""

    def __init__(self, in_channels: int = 6, style_dim: int = 128):
        super().__init__()
        # SA1: 2048→1024, r=0.1, K=32, 2 PVC(32,grid=32), MLP(32,32), no attn
        self.sa1 = SetAbstraction(
            in_ch=in_channels, out_ch=32, n_centers=1024,
            radius=0.1, K=32,
            n_pvc=2, pvc_hidden=32, pvc_res=32,
            mlp_dims=[32, 32],
            use_attention=False, style_dim=None,
        )
        # SA2: 1024→256, r=0.2, K=32, 1 PVC(32,grid=16), MLP(32,64), attn(128)
        self.sa2 = SetAbstraction(
            in_ch=32, out_ch=64, n_centers=256,
            radius=0.2, K=32,
            n_pvc=1, pvc_hidden=32, pvc_res=16,
            mlp_dims=[32, 64],
            use_attention=True, attn_dim=64, style_dim=None,
        )
        # Global max pool → (B, 64) → Linear → mu/logvar (style_dim=128)
        self.fc_mu     = nn.Linear(64, style_dim)
        self.fc_logvar = nn.Linear(64, style_dim)

    def forward(self, x: torch.Tensor):
        """x: (B, N, 6)"""
        xyz      = x[..., :3]
        xyz1, f1 = self.sa1(xyz, x)              # (B,1024,3), (B,1024,32)
        _,    f2 = self.sa2(xyz1, f1)            # (B,256,3),  (B,256,64)
        g = f2.max(dim=1).values                 # (B, 64)
        return self.fc_mu(g), self.fc_logvar(g)


# ---------------------------------------------------------------------------
# Local Encoder  (LION Table 6)
# ---------------------------------------------------------------------------

class LocalEncoder(nn.Module):
    """
    Per-point encoder that produces z_l of shape (B, N, 3 + latent_dim).
    Conditioned on z_g (global style) via AdaGN throughout.

    forward(x, style) — same signature as before.
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 3,
                 style_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_dim  = 3 + latent_dim

        # --- Encoder path ---
        # SA1: 2048→1024, r=0.1, K=32, 2 PVC(32,grid=32), MLP(32,32)
        self.sa1 = SetAbstraction(
            in_ch=in_channels, out_ch=32, n_centers=1024,
            radius=0.1, K=32,
            n_pvc=2, pvc_hidden=32, pvc_res=32,
            mlp_dims=[32, 32],
            use_attention=False, style_dim=style_dim,
        )
        # SA2: 1024→256, r=0.2, K=32, 1 PVC(64,grid=16), MLP(64,128), attn(128)
        self.sa2 = SetAbstraction(
            in_ch=32, out_ch=128, n_centers=256,
            radius=0.2, K=32,
            n_pvc=1, pvc_hidden=64, pvc_res=16,
            mlp_dims=[64, 128],
            use_attention=True, attn_dim=128, style_dim=style_dim,
        )
        # SA3: 256→64, r=0.4, K=32, 1 PVC(128,grid=8), MLP(128,256)
        self.sa3 = SetAbstraction(
            in_ch=128, out_ch=256, n_centers=64,
            radius=0.4, K=32,
            n_pvc=1, pvc_hidden=128, pvc_res=8,
            mlp_dims=[128, 256],
            use_attention=False, style_dim=style_dim,
        )
        # SA4: 64→16, r=0.8, K=32, no PVC, MLP(128,128,128)
        self.sa4 = SetAbstraction(
            in_ch=256, out_ch=128, n_centers=16,
            radius=0.8, K=32,
            n_pvc=0, pvc_hidden=128, pvc_res=8,
            mlp_dims=[128, 128, 128],
            use_attention=False, style_dim=style_dim,
        )
        # Global attention over 16 centres
        self.global_attn = PointSelfAttention(128, num_heads=4)

        # --- Decoder / Feature Propagation path ---
        # FP4: 16→64, concat(128+256)=384, MLP(128,128), 2 PVC(128,grid=8)
        self.fp4 = FeaturePropagation(
            in_ch_high=256, in_ch_skip=128, out_ch=128,
            mlp_dims=[128, 128],
            n_pvc=2, pvc_hidden=128, pvc_res=8,
            use_attention=False, style_dim=style_dim,
        )
        # FP3: 64→256, concat(128+128)=256, MLP(128,128), 2 PVC(128,grid=16)
        self.fp3 = FeaturePropagation(
            in_ch_high=128, in_ch_skip=128, out_ch=128,
            mlp_dims=[128, 128],
            n_pvc=2, pvc_hidden=128, pvc_res=16,
            use_attention=False, style_dim=style_dim,
        )
        # FP2: 256→1024, concat(128+32)=160, MLP(128,128), 3 PVC(128,grid=8), attn(128)
        self.fp2 = FeaturePropagation(
            in_ch_high=32, in_ch_skip=128, out_ch=128,
            mlp_dims=[128, 128],
            n_pvc=3, pvc_hidden=128, pvc_res=8,
            use_attention=True, attn_dim=128, style_dim=style_dim,
        )
        # FP1: 1024→2048, concat(128+in_channels), MLP(128,128,64), 2 PVC(64,grid=32)
        self.fp1 = FeaturePropagation(
            in_ch_high=in_channels, in_ch_skip=128, out_ch=64,
            mlp_dims=[128, 128, 64],
            n_pvc=2, pvc_hidden=64, pvc_res=32,
            use_attention=False, style_dim=style_dim,
        )

        # Output heads
        self.feat_proj  = nn.Linear(64, 128)
        self.dropout    = nn.Dropout(0.1)
        self.fc_out     = nn.Linear(128, self.total_dim * 2)

        # Bias init: logvar starts at -6 (small variance), mu starts at 0
        nn.init.constant_(self.fc_out.bias[self.total_dim:], -6.0)
        nn.init.zeros_(self.fc_out.weight[:self.total_dim])
        nn.init.zeros_(self.fc_out.bias[:self.total_dim])

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        x:     (B, N, in_channels)
        style: (B, style_dim)
        Returns: mu_l (B, N, total_dim), logvar_l (B, N, total_dim)
        """
        xyz = x[..., :3]

        # --- SA path: each SA does its own FPS + ball query internally ---
        xyz1, f1 = self.sa1(xyz,  x,  style)    # (B,1024,3), (B,1024,32)
        xyz2, f2 = self.sa2(xyz1, f1, style)    # (B,256,3),  (B,256,128)
        xyz3, f3 = self.sa3(xyz2, f2, style)    # (B,64,3),   (B,64,256)
        xyz4, f4 = self.sa4(xyz3, f3, style)    # (B,16,3),   (B,16,128)
        f4 = self.global_attn(f4)               # (B,16,128)

        # --- FP path ---
        _, fp4_feat = self.fp4(xyz3, f3,  xyz4, f4,       style)  # (B,64,128)
        _, fp3_feat = self.fp3(xyz2, f2,  xyz3, fp4_feat, style)  # (B,256,128)
        _, fp2_feat = self.fp2(xyz1, f1,  xyz2, fp3_feat, style)  # (B,1024,128)
        _, fp1_feat = self.fp1(xyz,  x,   xyz1, fp2_feat, style)  # (B,N,64)

        # --- Output ---
        h   = self.dropout(F.silu(self.feat_proj(fp1_feat)))   # (B,N,128)
        out = self.fc_out(h)                                    # (B,N,total_dim*2)

        mu_raw = out[..., :self.total_dim]
        logvar  = out[..., self.total_dim:]

        # Return raw delta — the position skip (xyz + 0.01 * delta) is applied
        # in Vae.forward so that KL is computed on the delta, not on xyz itself.
        return mu_raw, logvar


# ---------------------------------------------------------------------------
# Shape Encoder  (Real_latent — flat global bottleneck, no coordinate shortcut)
# ---------------------------------------------------------------------------

class ShapeEncoder(nn.Module):
    """
    Encodes a point cloud to a FLAT global shape latent z_l ∈ ℝ^{latent_size}.

    Uses the same 4-SA-block downsampling as LocalEncoder, but stops at SA4 and
    globally max-pools the 16 representative points into a single vector.
    No Feature Propagation, no per-point output, no coordinate shortcuts.

    NOT conditioned on z_g — encodes shape independently. The decoder receives
    both vectors and learns to use them complementarily.
    """

    def __init__(self, in_channels: int = 6, latent_size: int = 1024):
        super().__init__()
        self.sa1 = SetAbstraction(
            in_ch=in_channels, out_ch=32, n_centers=1024,
            radius=0.1, K=32,
            n_pvc=2, pvc_hidden=32, pvc_res=32,
            mlp_dims=[32, 32], use_attention=False, style_dim=None,
        )
        self.sa2 = SetAbstraction(
            in_ch=32, out_ch=128, n_centers=256,
            radius=0.2, K=32,
            n_pvc=1, pvc_hidden=64, pvc_res=16,
            mlp_dims=[64, 128], use_attention=True, attn_dim=128, style_dim=None,
        )
        self.sa3 = SetAbstraction(
            in_ch=128, out_ch=256, n_centers=64,
            radius=0.4, K=32,
            n_pvc=1, pvc_hidden=128, pvc_res=8,
            mlp_dims=[128, 256], use_attention=False, style_dim=None,
        )
        self.sa4 = SetAbstraction(
            in_ch=256, out_ch=128, n_centers=16,
            radius=0.8, K=32,
            n_pvc=0, pvc_hidden=128, pvc_res=8,
            mlp_dims=[128, 128, 128], use_attention=False, style_dim=None,
        )
        self.global_attn = PointSelfAttention(128, num_heads=4)

        self.fc_mu     = nn.Linear(128, latent_size)
        self.fc_logvar = nn.Linear(128, latent_size)
        nn.init.constant_(self.fc_logvar.bias, -6.0)

    def forward(self, x: torch.Tensor):
        """x: (B, N, in_channels)"""
        xyz = x[..., :3]
        xyz1, f1 = self.sa1(xyz,  x)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)
        _,    f4 = self.sa4(xyz3, f3)
        f4 = self.global_attn(f4)      # (B, 16, 128)
        g  = f4.max(dim=1).values      # (B, 128)
        return self.fc_mu(g), self.fc_logvar(g)
