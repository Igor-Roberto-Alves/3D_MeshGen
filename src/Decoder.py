import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import AdaGN, SqueezeExcitation


class SharedMLP(nn.Module):
    """
    Point-wise MLP (1x1 Conv1d) operating on (B, C, N).

    BatchNorm1d was replaced by AdaGN: when ``style_dim`` is given each layer is
    normalised with Adaptive GroupNorm conditioned on the global style z_g
    (matches the rest of the decoder). When ``style_dim`` is None it falls back
    to plain GroupNorm. This removes the small-batch instability of BatchNorm.
    """
    def __init__(self, channels, style_dim: int | None = None, act: bool = True):
        super().__init__()
        self.style_dim = style_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act   = nn.GELU() if act else nn.Identity()
        for i in range(len(channels) - 1):
            cout = channels[i + 1]
            self.convs.append(nn.Conv1d(channels[i], cout, 1, bias=False))
            if style_dim is None:
                self.norms.append(nn.GroupNorm(min(8, cout), cout))
            else:
                self.norms.append(AdaGN(cout, style_dim, num_groups=min(8, cout)))

    def forward(self, x: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, N)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x, style) if self.style_dim is not None else norm(x)
            x = self.act(x)
        return x


class VoxelBranch(nn.Module):
    def __init__(self, feat_channels, out_channels, resolution=16):
        super().__init__()
        self.resolution = resolution
        mid = max(out_channels * 2, 8)
        self.vox_conv1 = nn.Conv3d(feat_channels, mid, 3, padding=1, bias=False)
        self.vox_gn1   = nn.GroupNorm(min(8, mid), mid)
        self.vox_conv2 = nn.Conv3d(mid, out_channels, 3, padding=1, bias=False)
        self.vox_gn2   = nn.GroupNorm(min(8, out_channels), out_channels)
        self.vox_se    = SqueezeExcitation(out_channels)

    def _voxelise(self, coords, features, R):
        B, C, N = features.shape
        device = features.device
        idx = coords.long().clamp(0, R - 1)
        flat_idx = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]
        # float32 accumulation prevents float16 overflow under AMP
        feat32  = features.float()
        voxels  = torch.zeros(B, C, R * R * R, device=device, dtype=torch.float32)
        count   = torch.zeros(B, 1, R * R * R, device=device, dtype=torch.float32)
        voxels.scatter_add_(2, flat_idx.unsqueeze(1).expand_as(feat32), feat32)
        count.scatter_add_(2, flat_idx.unsqueeze(1),
                           torch.ones(B, 1, N, device=device, dtype=torch.float32))
        return (voxels / count.clamp(min=1.0)).view(B, C, R, R, R).to(features.dtype)

    def forward(self, xyz, feat):
        B, N, _ = xyz.shape
        R = self.resolution
        coords = xyz.permute(0, 2, 1)
        feats  = feat.permute(0, 2, 1)
        norm_coords = (coords + 1.0) * 0.5 * (R - 1)
        norm_coords = norm_coords.clamp(0.0, float(R - 1))
        voxel_grid  = self._voxelise(norm_coords, feats, R)
        vx = F.gelu(self.vox_gn1(self.vox_conv1(voxel_grid)))
        vx = self.vox_gn2(self.vox_conv2(vx))
        # SE operates on (B, C, N) — flatten spatial dims to N for SE, then restore
        B2, C2, R2, _, _ = vx.shape
        vx_flat = vx.view(B2, C2, -1)          # (B, C, R^3)
        vx_flat = self.vox_se(vx_flat)
        voxel_feats = vx_flat.view(B2, C2, R2, R2, R2)
        sample_grid = (norm_coords / (R - 1) * 2 - 1).clamp(-1.0, 1.0)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        sample_grid = sample_grid.expand(-1, 1, 1, N, 3)
        sampled = F.grid_sample(voxel_feats, sample_grid,
                                mode='bilinear', align_corners=True, padding_mode='border')
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)


class PointBranch(nn.Module):
    def __init__(self, feat_channels, out_channels, style_dim: int | None = None):
        super().__init__()
        self.mlp = SharedMLP([feat_channels, out_channels * 2, out_channels],
                             style_dim=style_dim)

    def forward(self, xyz, feat, style=None):
        return self.mlp(feat.permute(0, 2, 1), style).permute(0, 2, 1)


class PVConvBlockDecoder(nn.Module):
    """PVCNN block with AdaGN conditioning — used inside the decoder."""
    def __init__(self, feat_in, feat_out, style_dim, resolution=16, dropout=0.1):
        super().__init__()
        half = feat_out // 2
        self.voxel   = VoxelBranch(feat_in, half, resolution)
        self.point   = PointBranch(feat_in, half, style_dim=style_dim)
        self.adagn   = AdaGN(num_channels=feat_out, style_dim=style_dim, num_groups=8)
        self.act     = nn.GELU()
        self.conv    = nn.Conv1d(feat_out, feat_out, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xyz, feat, style):
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat, style)
        x = torch.cat([v, p], dim=2).permute(0, 2, 1)  # (B, feat_out, N)
        x = self.adagn(x, style)
        return self.dropout(self.conv(self.act(x))).permute(0, 2, 1)  # (B, N, feat_out)


# ---------------------------------------------------------------------------
# LION Decoder (correct architecture)
# ---------------------------------------------------------------------------

class LIONDecoder(nn.Module):
    """
    Point-cloud decoder following the real LION architecture (LatentPointDecPVC).

    z_l has shape (B, N, 3 + latent_dim) matching the encoder output:
      - z_l[:, :, :3]        — position latents (used directly as xyz_init)
      - z_l[:, :, 3:]        — feature latents  (processed through PVCNN)

    The decoder extracts xyz_init = tanh(z_l[:,:,:3]).  tanh bounds coordinates
    to [-1,1] for voxelisation stability (our VoxelBranch assumes this range).
    In LION's PointNet++ SA blocks this is not needed; here it is a pragmatic fix.

    Skip connection (from LION latent_points_ada.py):
        xyz_out = output_head(PVCNN_features) + xyz_init
    i.e. the PVCNN predicts a residual correction over the latent positions.

    At epoch 0 the output_head is near-zero (small init), so:
        xyz_out ≈ xyz_init ≈ tanh(mu_l[:,:,:3]) ≈ mu_l[:,:,:3] ≈ input_xyz
    giving identity-like reconstruction from the very first forward pass.
    """

    def __init__(self, latent_dim: int = 3, style_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # feat_proj maps the FEATURE part of z_l (latent_dim channels) to 256-dim.
        # The POSITION part (first 3 channels) is used directly as xyz_init.
        # Conditioned on z_g via AdaGN (no BatchNorm).
        self.feat_proj = SharedMLP([latent_dim, 128, 256], style_dim=style_dim)

        self.stage0 = PVConvBlockDecoder(256, 256, style_dim, resolution=16)
        self.stage1 = PVConvBlockDecoder(256, 128, style_dim, resolution=32)
        self.stage2 = PVConvBlockDecoder(128, 64,  style_dim, resolution=32)

        # Position residual head: near-zero init so delta≈0 at epoch 0
        # → xyz_out ≈ xyz_init (identity through latent positions).
        self.output_head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1),
        )
        nn.init.normal_(self.output_head[-1].weight, std=0.01)
        nn.init.zeros_(self.output_head[-1].bias)

        # Normal prediction head — separate branch, same small-init strategy.
        self.output_head_normals = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1),
        )
        nn.init.normal_(self.output_head_normals[-1].weight, std=0.01)
        nn.init.zeros_(self.output_head_normals[-1].bias)

    def forward(self, z_l: torch.Tensor, z_global: torch.Tensor):
        """
        z_l      : (B, N, 3 + latent_dim)  — stochastic per-point latent
        z_global : (B, style_dim)           — global shape context

        Returns
        -------
        xyz_out  (B, N, 3)   reconstructed positions (normalised scale)
        nrm_out  (B, N, 3)   predicted unit normals
        """
        # Split z_l into position and feature parts (LION LatentPointDecPVC logic).
        xyz_latent = z_l[:, :, :3]                         # (B, N, 3) position latents
        feat_raw   = z_l[:, :, 3:]                         # (B, N, latent_dim)

        # Use position latents directly — tanh was removed because normalize_pc
        # maps the farthest point to exactly ±1.0, which would require mu→±∞
        # to represent through tanh, causing the encoder to diverge (kl_pos→∞).
        # The VoxelBranch already clamps out-of-range coords safely.
        xyz_init = xyz_latent                              # (B, N, 3)

        # Project feature latents to PVCNN feature dimension (AdaGN on z_global).
        z_f  = feat_raw.permute(0, 2, 1)                   # (B, latent_dim, N)
        feat = self.feat_proj(z_f, z_global).permute(0, 2, 1)  # (B, N, 256)

        # PVCNN: uses xyz_init for spatial structure (voxelisation reference)
        feat = self.stage0(xyz_init, feat, z_global)
        feat = self.stage1(xyz_init, feat, z_global)
        feat = self.stage2(xyz_init, feat, z_global)

        feat_t  = feat.permute(0, 2, 1)                    # (B, 64, N)
        delta   = self.output_head(feat_t).permute(0, 2, 1)         # (B, N, 3)
        nrm_raw = self.output_head_normals(feat_t).permute(0, 2, 1) # (B, N, 3)

        # LION skip: final positions = residual correction + latent positions
        xyz_out = delta + xyz_init
        nrm_out = F.normalize(nrm_raw, dim=-1)
        return xyz_out, nrm_out


# ---------------------------------------------------------------------------
# Flat Decoder  (Real_latent — generates N points from flat (z_l, z_g))
# ---------------------------------------------------------------------------

class FlatDecoder(nn.Module):
    """
    Generates a point cloud from (z_l ∈ ℝ^{latent_size}, z_g ∈ ℝ^{style_dim}).

    No coordinate shortcut — z_l and z_g are the sole information source.

    Architecture
    ------------
    1. Broadcast z_l to N points via 1×1 Conv1d (every point sees the full latent).
    2. Learnable seed_xyz provides the spatial reference for PVCNN voxelisation.
       Initialised on the unit sphere so voxelisation is non-degenerate at epoch 0.
    3. Three PVCNN stages conditioned on z_g (AdaGN).
    4. Output head predicts a delta over seed_xyz → final positions.
    """

    def __init__(
        self,
        latent_size: int,
        style_dim:   int,
        num_points:  int = 2048,
        hidden:      int = 256,
    ):
        super().__init__()
        self.num_points = num_points

        self.latent_expand = nn.Sequential(
            nn.Conv1d(latent_size, hidden, 1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
        )

        seed = torch.randn(1, num_points, 3)
        seed = seed / seed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.seed_xyz = nn.Parameter(seed)

        self.stage0 = PVConvBlockDecoder(hidden,      hidden,      style_dim, resolution=8)
        self.stage1 = PVConvBlockDecoder(hidden,      hidden // 2, style_dim, resolution=16)
        self.stage2 = PVConvBlockDecoder(hidden // 2, 64,          style_dim, resolution=32)

        self.output_head = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.GELU(),
            nn.Conv1d(64,  3, 1),
        )
        nn.init.normal_(self.output_head[-1].weight, std=0.01)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, z_l: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """
        z_l : (B, latent_size)
        z_g : (B, style_dim)
        Returns xyz_out (B, N, 3)
        """
        B = z_l.shape[0]
        N = self.num_points

        z_l_exp = z_l.unsqueeze(-1).expand(-1, -1, N)          # (B, latent_size, N)
        feat    = self.latent_expand(z_l_exp).permute(0, 2, 1)  # (B, N, hidden)

        xyz = self.seed_xyz.expand(B, -1, -1)                   # (B, N, 3)

        feat = self.stage0(xyz, feat, z_g)
        feat = self.stage1(xyz, feat, z_g)
        feat = self.stage2(xyz, feat, z_g)

        delta = self.output_head(feat.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 3)
        return xyz + delta
