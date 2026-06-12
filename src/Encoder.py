import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *


class SharedMLP(nn.Sequential):
    """1-D shared MLP applied point-wise (B, C, N) → (B, C', N)."""
    def __init__(self, channels: list[int], bn: bool = True, act: bool = True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            if act:
                layers.append(nn.GELU())
        super().__init__(*layers)


# ===========================================================================
# Voxel branch  — receives XYZ and features SEPARATELY
# ===========================================================================
class VoxelBranch(nn.Module):
    """
    Voxelises a point cloud using EXPLICIT XYZ coords (always in [-1,1]),
    applies 3-D convolutions on the FEATURES, then samples back.

    THE ROOT CAUSE OF THE CUBE BUG:
    The previous API took a single tensor and assumed points[:,:,:3] == XYZ.
    After the first PVConvBlock the tensor has C=64/128/256 channels where
    the first 3 are arbitrary learned features — NOT coordinates.
    Every downstream voxelisation was placing all points in the wrong voxels,
    so the 3-D conv saw a random scrambled grid every forward pass.

    Fix: XYZ coords are now passed explicitly and never mixed with features.
    """
    def __init__(self, feat_channels: int, out_channels: int, resolution: int = 16):
        super().__init__()
        self.resolution = resolution
        mid = out_channels * 2

        self.voxel_net = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, mid), num_channels=mid),
            nn.GELU(),
            nn.Conv3d(mid, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def _voxelise(self, coords: torch.Tensor, features: torch.Tensor, R: int) -> torch.Tensor:
        """
        coords   : (B, 3, N)  already in [0, R-1]
        features : (B, C, N)
        """
        B, C, N = features.shape
        device = features.device

        idx      = coords.long().clamp(0, R - 1)
        flat_idx = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]  # (B, N)

        voxels = torch.zeros(B, C, R * R * R, device=device, dtype=features.dtype)
        count  = torch.zeros(B, 1, R * R * R, device=device, dtype=features.dtype)

        voxels.scatter_add_(2, flat_idx.unsqueeze(1).expand_as(features), features)
        count.scatter_add_(2, flat_idx.unsqueeze(1),
                           torch.ones(B, 1, N, device=device, dtype=features.dtype))

        voxels = voxels / count.clamp(min=1.0)
        return voxels.view(B, C, R, R, R)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        xyz  : (B, N, 3)  — coordinates in [-1, 1]   (NEVER mixed with features)
        feat : (B, N, C)  — learned features
        returns (B, N, out_channels)
        """
        B, N, _ = xyz.shape
        R = self.resolution

        coords = xyz.permute(0, 2, 1)               # (B, 3, N)
        feats  = feat.permute(0, 2, 1)              # (B, C, N)

        # Fixed mapping [-1,1] → [0, R-1]
        norm_coords = (coords + 1.0) * 0.5 * (R - 1)
        norm_coords = torch.clamp(norm_coords, 0.0, float(R - 1))

        voxel_grid  = self._voxelise(norm_coords, feats, R)
        voxel_feats = self.voxel_net(voxel_grid)

        # sample_grid back to [-1,1]
        sample_grid = norm_coords / (R - 1) * 2 - 1
        sample_grid = torch.clamp(sample_grid, -1.0, 1.0)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        sample_grid = sample_grid.expand(-1, 1, 1, N, 3)

        sampled = F.grid_sample(
            voxel_feats, sample_grid,
            mode="bilinear", align_corners=True, padding_mode="border"
        )
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)   # (B, N, out_channels)


# ===========================================================================
# Point branch  — also receives XYZ + features separately (API consistency)
# ===========================================================================
class PointBranch(nn.Module):
    def __init__(self, feat_channels: int, out_channels: int):
        super().__init__()
        self.mlp = SharedMLP([feat_channels, out_channels * 2, out_channels])

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """xyz: (B,N,3) ignored here (point-wise MLP needs no coords), feat: (B,N,C)"""
        return self.mlp(feat.permute(0, 2, 1)).permute(0, 2, 1)


# ===========================================================================
# PVConvBlock  — explicit (xyz, feat) API
# ===========================================================================
class PVConvBlock(nn.Module):
    """
    PVCNN-style block with explicit XYZ / feature separation.

    Args:
        feat_in   : input feature channels
        feat_out  : output feature channels
        resolution: voxel grid size

    forward(xyz, feat) → feat_out
        xyz  : (B, N, 3)  coordinates — passed through unchanged, never modified
        feat : (B, N, feat_in)
    returns:
        (B, N, feat_out)  — new features; xyz is NOT returned (caller keeps it)
    """
    def __init__(self, feat_in: int, feat_out: int, resolution: int = 16):
        super().__init__()
        half = feat_out // 2
        self.voxel = VoxelBranch(feat_in, half, resolution)
        self.point = PointBranch(feat_in, half)

        self.fuse = nn.Sequential(
            nn.Conv1d(feat_out, feat_out, 1, bias=False),
            nn.BatchNorm1d(feat_out),
            nn.GELU(),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat)
        x = torch.cat([v, p], dim=2)                          # (B, N, feat_out)
        return self.fuse(x.permute(0, 2, 1)).permute(0, 2, 1)


# ===========================================================================
# EncoderStyle  (global style vector z0)
# ===========================================================================
class EncoderStyle(nn.Module):
    def __init__(self, in_channels: int = 6, style_dim: int = 512):
        super().__init__()
        # First stage takes full input (XYZ + normals) as features
        # XYZ is extracted separately and kept fixed across all stages
        self.input_proj = SharedMLP([in_channels, 64])

        self.stage0 = PVConvBlock(64,  128, resolution=32)
        self.stage1 = PVConvBlock(128, 256, resolution=16)
        self.stage2 = PVConvBlock(256, 256, resolution=8)

        self.fc_mu = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, style_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, style_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, 6)  XYZ in [-1,1] + normals"""
        xyz  = x[:, :, :3]                          # (B, N, 3) — fixed throughout
        feat = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 64)

        feat = self.stage0(xyz, feat)                # (B, N, 128)
        feat = self.stage1(xyz, feat)                # (B, N, 256)
        feat = self.stage2(xyz, feat)                # (B, N, 256)

        g      = feat.max(dim=1).values              # (B, 256)
        mu     = self.fc_mu(g)
        logvar = self.fc_logvar(g)
        return mu, logvar

    def z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


# ===========================================================================
# Encoder  (local latent point cloud)
# ===========================================================================

class Encoder(nn.Module):
    def __init__(
            self,
            latent_dim=256,
            in_channels=6,
            style_dim=512,
            num_latent_points=512,
        ):
            super().__init__()

            self.num_latent_points = num_latent_points

            self.input_proj = SharedMLP([in_channels, 64])

            self.stage0 = PVConvBlock(64, 128, resolution=32)
            self.stage1 = PVConvBlock(128, 256, resolution=16)
            self.stage2 = PVConvBlock(256, 256, resolution=8)

            self.adagn = AdaGN(
                num_channels=256,
                style_dim=style_dim,
                num_groups=32
            )

            self.local_mlp = SharedMLP([256, 512])

            # --------------------------------------------------
            # XYZ latente (LION-style)
            # --------------------------------------------------

            self.fc_xyz_delta_mu = nn.Conv1d(
                512,
                3,
                1
            )

            self.fc_xyz_logvar = nn.Conv1d(
                512,
                3,
                1
            )

            # --------------------------------------------------
            # Features latentes
            # --------------------------------------------------

            self.fc_feat_mu = nn.Conv1d(
                512,
                latent_dim,
                1
            )

            self.fc_feat_logvar = nn.Conv1d(
                512,
                latent_dim,
                1
            )

    def forward(
            self,
            x,
            style
        ):

            xyz = x[:, :, :3]

            feat = self.input_proj(
                x.permute(0, 2, 1)
            ).permute(0, 2, 1)

            feat = self.stage0(
                xyz,
                feat
            )

            feat = self.stage1(
                xyz,
                feat
            )

            feat = self.stage2(
                xyz,
                feat
            )

            feat = self.adagn(
                feat.permute(0, 2, 1),
                style
            ).permute(0, 2, 1)

            # --------------------------------------------------
            # FPS
            # --------------------------------------------------

            fps_idx = farthest_point_sample(
                xyz,
                self.num_latent_points
            )

            latent_xyz = index_points(
                xyz,
                fps_idx
            )

            feat_lat = index_points(
                feat,
                fps_idx
            )

            feat_lat = self.local_mlp(
                feat_lat.permute(0, 2, 1)
            )

            # --------------------------------------------------
            # XYZ Gaussian (LION-like)
            # --------------------------------------------------

            delta_xyz_mu = self.fc_xyz_delta_mu(
                feat_lat
            ).permute(0, 2, 1)

            xyz_logvar = self.fc_xyz_logvar(
                feat_lat
            ).permute(0, 2, 1)

            xyz_logvar = torch.clamp(
                xyz_logvar,
                min=-8.0,
                max=4.0
            )

            # residual prediction around FPS anchors

            xyz_mu = latent_xyz + 0.1 * torch.tanh(
                delta_xyz_mu
            )

            # --------------------------------------------------
            # Feature Gaussian
            # --------------------------------------------------

            feat_mu = self.fc_feat_mu(
                feat_lat
            ).permute(0, 2, 1)

            feat_logvar = self.fc_feat_logvar(
                feat_lat
            ).permute(0, 2, 1)

            feat_logvar = torch.clamp(
                feat_logvar,
                min=-8.0,
                max=8.0
            )

            # --------------------------------------------------
            # Full latent distribution
            # --------------------------------------------------

            mu = torch.cat(
                [
                    xyz_mu,
                    feat_mu
                ],
                dim=-1
            )

            logvar = torch.cat(
                [
                    xyz_logvar,
                    feat_logvar
                ],
                dim=-1
            )

            return mu, logvar