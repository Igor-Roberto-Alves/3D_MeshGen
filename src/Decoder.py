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
        B, N, _ = xyz.shape
        R = self.resolution
        coords = xyz.permute(0, 2, 1)
        feats  = feat.permute(0, 2, 1)
        norm_coords = (coords + 1.0) * 0.5 * (R - 1)
        norm_coords = norm_coords.clamp(0.0, float(R - 1))
        voxel_grid  = self._voxelise(norm_coords, feats, R)
        voxel_feats = self.voxel_net(voxel_grid)
        sample_grid = (norm_coords / (R - 1) * 2 - 1).clamp(-1.0, 1.0)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        sample_grid = sample_grid.expand(-1, 1, 1, N, 3)
        sampled = F.grid_sample(voxel_feats, sample_grid,
                                mode='bilinear', align_corners=True, padding_mode='border')
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)


class PointBranch(nn.Module):
    def __init__(self, feat_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP([feat_channels, out_channels * 2, out_channels])

    def forward(self, xyz, feat):
        return self.mlp(feat.permute(0, 2, 1)).permute(0, 2, 1)


class PVConvBlockDecoder(nn.Module):
    """PVCNN block with AdaGN conditioning — used inside the decoder."""
    def __init__(self, feat_in, feat_out, style_dim, resolution=16):
        super().__init__()
        half = feat_out // 2
        self.voxel = VoxelBranch(feat_in, half, resolution)
        self.point = PointBranch(feat_in, half)
        self.adagn = AdaGN(num_channels=feat_out, style_dim=style_dim, num_groups=8)
        self.act   = nn.GELU()
        self.conv  = nn.Conv1d(feat_out, feat_out, 1)

    def forward(self, xyz, feat, style):
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat)
        x = torch.cat([v, p], dim=2).permute(0, 2, 1)  # (B, feat_out, N)
        x = self.adagn(x, style)
        return self.conv(self.act(x)).permute(0, 2, 1)  # (B, N, feat_out)


# ---------------------------------------------------------------------------
# LION Decoder
# ---------------------------------------------------------------------------

class LIONDecoder(nn.Module):
    """
    Point-cloud decoder conditioned on global shape latent z_g.

    Input:  z_local  (B, N, 3 + latent_dim)  — xyz anchors | per-point features
            z_global (B, style_dim)           — global shape context
    Output: xyz_out  (B, N, 3)               — refined point positions

    The decoder does NOT upsample. N_out == N_in; the VAE keeps the same
    point count throughout, matching the LION paper (Sec. 3).
    """
    def __init__(self, latent_dim=3, style_dim=256):
        super().__init__()
        # Project the small per-point latent to a richer working space
        self.feat_proj = SharedMLP([latent_dim, 128, 256])

        # Three PVCNN stages, all conditioned on z_g via AdaGN
        self.stage0 = PVConvBlockDecoder(256, 256, style_dim, resolution=16)
        self.stage1 = PVConvBlockDecoder(256, 128, style_dim, resolution=32)
        self.stage2 = PVConvBlockDecoder(128, 64,  style_dim, resolution=32)

        # Predict a small delta_xyz on top of the anchor positions
        self.output_head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1),
        )

    def forward(self, z_local, z_global):
        # z_local:  (B, N, 3 + latent_dim)
        # z_global: (B, style_dim)
        xyz  = z_local[..., :3]                               # anchor positions
        feat = z_local[..., 3:]                               # (B, N, latent_dim)

        feat = self.feat_proj(feat.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 256)
        feat = self.stage0(xyz, feat, z_global)
        feat = self.stage1(xyz, feat, z_global)
        feat = self.stage2(xyz, feat, z_global)

        delta = self.output_head(feat.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 3)

        # Weighted skip: anchors carry the coarse structure; delta refines it
        return xyz + 0.1 * delta
