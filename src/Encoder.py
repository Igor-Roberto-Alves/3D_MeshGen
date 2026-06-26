import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import AdaGN


class SharedMLP(nn.Sequential):
    def __init__(self, channels, bn=True, act=True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            if act:
                layers.append(nn.GELU())
        super().__init__(*layers)


class VoxelBranch(nn.Module):
    def __init__(self, feat_channels, out_channels, resolution=16):
        super().__init__()
        self.resolution = resolution
        mid = max(out_channels * 2, 8)
        self.voxel_net = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.GELU(),
            nn.Conv3d(mid, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )

    def _voxelise(self, coords, features, R):
        B, C, N = features.shape
        device = features.device
        idx = coords.long().clamp(0, R - 1)
        flat_idx = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]
        voxels = torch.zeros(B, C, R * R * R, device=device, dtype=features.dtype)
        count  = torch.zeros(B, 1, R * R * R, device=device, dtype=features.dtype)
        voxels.scatter_add_(2, flat_idx.unsqueeze(1).expand_as(features), features)
        count.scatter_add_(2, flat_idx.unsqueeze(1),
                           torch.ones(B, 1, N, device=device, dtype=features.dtype))
        return (voxels / count.clamp(min=1.0)).view(B, C, R, R, R)

    def forward(self, xyz, feat):
        # xyz: (B, N, 3) in [-1, 1];  feat: (B, N, C)
        B, N, _ = xyz.shape
        R = self.resolution
        coords = xyz.permute(0, 2, 1)            # (B, 3, N)
        feats  = feat.permute(0, 2, 1)           # (B, C, N)
        norm_coords = (coords + 1.0) * 0.5 * (R - 1)
        norm_coords = norm_coords.clamp(0.0, float(R - 1))
        voxel_grid  = self._voxelise(norm_coords, feats, R)
        voxel_feats = self.voxel_net(voxel_grid)
        sample_grid = (norm_coords / (R - 1) * 2 - 1).clamp(-1.0, 1.0)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        sample_grid = sample_grid.expand(-1, 1, 1, N, 3)
        sampled = F.grid_sample(voxel_feats, sample_grid,
                                mode='bilinear', align_corners=True, padding_mode='border')
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # (B, N, out_channels)


class PointBranch(nn.Module):
    def __init__(self, feat_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP([feat_channels, out_channels * 2, out_channels])

    def forward(self, xyz, feat):
        return self.mlp(feat.permute(0, 2, 1)).permute(0, 2, 1)


class PVConvBlock(nn.Module):
    """Plain PVCNN block — no style conditioning. Used in GlobalEncoder."""
    def __init__(self, feat_in, feat_out, resolution=16, dropout=0.1):
        super().__init__()
        half = feat_out // 2
        self.voxel = VoxelBranch(feat_in, half, resolution)
        self.point = PointBranch(feat_in, half)
        self.fuse  = nn.Sequential(
            nn.Conv1d(feat_out, feat_out, 1, bias=False),
            nn.BatchNorm1d(feat_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, xyz, feat):
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat)
        x = torch.cat([v, p], dim=2)
        return self.fuse(x.permute(0, 2, 1)).permute(0, 2, 1)


class PVConvBlockConditioned(nn.Module):
    """PVCNN block with AdaGN style conditioning. Used in LocalEncoder."""
    def __init__(self, feat_in, feat_out, style_dim, resolution=16, dropout=0.1):
        super().__init__()
        half = feat_out // 2
        self.voxel   = VoxelBranch(feat_in, half, resolution)
        self.point   = PointBranch(feat_in, half)
        self.adagn   = AdaGN(num_channels=feat_out, style_dim=style_dim, num_groups=8)
        self.act     = nn.GELU()
        self.conv    = nn.Conv1d(feat_out, feat_out, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xyz, feat, style):
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat)
        x = torch.cat([v, p], dim=2).permute(0, 2, 1)  # (B, feat_out, N)
        x = self.adagn(x, style)
        return self.dropout(self.conv(self.act(x))).permute(0, 2, 1)  # (B, N, feat_out)


# ---------------------------------------------------------------------------
# Global Encoder  (LION paper: "shape-level VAE encoder")
# ---------------------------------------------------------------------------

class GlobalEncoder(nn.Module):
    """
    Encodes the full point cloud into a single global shape latent z_g.

    Architecture (LION Sec. 3):
        PVCNN backbone  →  global max-pool  →  fc_mu / fc_logvar
    Returns mu_g, logvar_g  each of shape (B, style_dim).
    """
    def __init__(self, in_channels=6, style_dim=256):
        super().__init__()
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64,          128, resolution=16)
        self.stage2 = PVConvBlock(128,         256, resolution=8)
        self.fc_mu     = nn.Linear(256, style_dim)
        self.fc_logvar = nn.Linear(256, style_dim)

    def forward(self, x):
        # x: (B, N, in_channels) with xyz already in [-1, 1]
        xyz  = x[..., :3]
        feat = x
        feat = self.stage0(xyz, feat)
        feat = self.stage1(xyz, feat)
        feat = self.stage2(xyz, feat)
        g = feat.max(dim=1).values          # global max-pool → (B, 256)
        return self.fc_mu(g), self.fc_logvar(g)


# ---------------------------------------------------------------------------
# Local Encoder  (LION paper: "point-level VAE encoder")
# ---------------------------------------------------------------------------

class LocalEncoder(nn.Module):
    """
    Encodes per-point latent features conditioned on the global shape z_g.

    latent_dim is fixed at 3 (LION paper D.1): mu_l represents a 3-channel
    position perturbation. The skip z_l = xyz + 0.01 * mu_l is applied in
    Vae.forward, so z_l lives in the same 3D position space as the input.
    """
    def __init__(self, in_channels=6, latent_dim=3, style_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.stage0 = PVConvBlockConditioned(in_channels, 64,  style_dim, resolution=32)
        self.stage1 = PVConvBlockConditioned(64,          128, style_dim, resolution=16)
        self.stage2 = PVConvBlockConditioned(128,         256, style_dim, resolution=8)
        self.fc_mu     = nn.Conv1d(256, latent_dim, 1)
        self.fc_logvar = nn.Conv1d(256, latent_dim, 1)
        # Paper D.1 — variance scaling: bias -6 pushes posterior std ≈ 0 at init
        # so z_l ≈ xyz_anchor (identity mapping through encoder).
        nn.init.constant_(self.fc_logvar.bias, -6.0)
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)

    def forward(self, x, style):
        # x: (B, N, in_channels);  style: (B, style_dim)
        xyz  = x[..., :3]
        feat = x
        feat = self.stage0(xyz, feat, style)
        feat = self.stage1(xyz, feat, style)
        feat = self.stage2(xyz, feat, style)
        f = feat.permute(0, 2, 1)                    # (B, 256, N)
        mu     = self.fc_mu(f).permute(0, 2, 1)      # (B, N, latent_dim)
        logvar = self.fc_logvar(f).permute(0, 2, 1)  # (B, N, latent_dim)
        return mu, logvar
