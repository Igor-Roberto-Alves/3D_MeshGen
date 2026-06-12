import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GlobalAttention(nn.Module):
    """Global Self-Attention bottleneck mapped after the latent space."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, C)
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)


class AdaGN(nn.Module):
    """Official LION Adaptive Group Normalization module."""
    def __init__(self, num_channels: int, style_dim: int, num_groups: int = 8):
        super().__init__()
        self.num_groups = num_groups
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=False)
        self.fc = nn.Linear(style_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, N) | style shape: (B, style_dim)
        x_norm = self.gn(x)
        
        # Predict scale (gamma) and bias (beta) from global shape style
        style_effects = self.fc(style).unsqueeze(-1)  # (B, 2*C, 1)
        gamma, beta = torch.chunk(style_effects, 2, dim=1)
        
        return x_norm * (1 + gamma) + beta


# ===========================================================================
# PVConv Components (Imported directly inside the file for standalone safety)
# ===========================================================================

class VoxelBranch(nn.Module):
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
        B, N, _ = xyz.shape
        R = self.resolution

        coords = xyz.permute(0, 2, 1)               # (B, 3, N)
        feats  = feat.permute(0, 2, 1)              # (B, C, N)

        norm_coords = (coords + 1.0) * 0.5 * (R - 1)
        norm_coords = torch.clamp(norm_coords, 0.0, float(R - 1))

        voxel_grid  = self._voxelise(norm_coords, feats, R)
        voxel_feats = self.voxel_net(voxel_grid)

        sample_grid = norm_coords / (R - 1) * 2 - 1
        sample_grid = torch.clamp(sample_grid, -1.0, 1.0)
        sample_grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        sample_grid = sample_grid.expand(-1, 1, 1, N, 3)

        sampled = F.grid_sample(
            voxel_feats, sample_grid,
            mode="bilinear", align_corners=True, padding_mode="border"
        )
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)   # (B, N, out_channels)


class PointBranch(nn.Module):
    def __init__(self, feat_channels: int, out_channels: int):
        super().__init__()
        self.mlp = SharedMLP([feat_channels, out_channels * 2, out_channels])

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat.permute(0, 2, 1)).permute(0, 2, 1)


class PVConvBlockDecoder(nn.Module):
    """PVConv block for the Decoder that encapsulates AdaGN modulation natively."""
    def __init__(self, feat_in: int, feat_out: int, style_dim: int, resolution: int = 16):
        super().__init__()
        half = feat_out // 2
        self.voxel = VoxelBranch(feat_in, half, resolution)
        self.point = PointBranch(feat_in, half)

        self.adagn = AdaGN(num_channels=feat_out, style_dim=style_dim, num_groups=8)
        self.act = nn.GELU()
        self.conv_out = nn.Conv1d(feat_out, feat_out, 1, bias=True)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        v = self.voxel(xyz, feat)
        p = self.point(xyz, feat)
        x = torch.cat([v, p], dim=2)                          # (B, N, feat_out)
        
        # Style modulation happens internally over the fused representations
        x_modulated = self.adagn(x.permute(0, 2, 1), style)   # (B, feat_out, N)
        x_out = self.conv_out(self.act(x_modulated)).permute(0, 2, 1)
        return x_out


# ===========================================================================
# Density / Upsampling Head
# ===========================================================================

class PointUpsampleBlock(nn.Module):
    """N → k*N via learned coordinate offsets."""
    def __init__(self, feat_channels: int, k: int = 4):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_channels, feat_channels, 1),
            nn.GELU(),
            nn.Conv1d(feat_channels, 3 * k, 1),
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        offsets = self.mlp(feat.permute(0, 2, 1))                      # (B, 3k, N)
        offsets = offsets.view(B, self.k, 3, N).permute(0, 3, 1, 2)   # (B, N, k, 3)
        new_xyz = xyz.unsqueeze(2) + offsets                           # (B, N, k, 3)
        return new_xyz.reshape(B, N * self.k, 3)


# ===========================================================================
# Core LION Decoder Structure
# ===========================================================================

class LIONDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        style_dim=512,
        input_dim=3,
        up_factor = 0,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.feat_proj = SharedMLP([latent_dim, 256])

        self.stage0 = PVConvBlockDecoder(
            256, 256,
            style_dim=style_dim,
            resolution=8
        )

        self.stage1 = PVConvBlockDecoder(
            256, 128,
            style_dim=style_dim,
            resolution=16
        )

        self.stage2 = PVConvBlockDecoder(
            128, 128,
            style_dim=style_dim,
            resolution=32
        )

        self.coord_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1)
        )
        self.output_head = nn.Sequential(
        nn.Conv1d(128, 128, 1),
        nn.GELU(),
        nn.Conv1d(128, 64, 1),
        nn.GELU(),
        nn.Conv1d(64, 6, 1)
    )

    def forward(self, latent_points, style):

        # latent_points: (B, N, 3 + latent_dim)

        xyz = latent_points[:, :, :3].contiguous()
        feat = latent_points[:, :, 3:]

        feat = self.feat_proj(
            feat.permute(0, 2, 1)
        ).permute(0, 2, 1)

        feat = self.stage0(xyz, feat, style)
        feat = self.stage1(xyz, feat, style)
        feat = self.stage2(xyz, feat, style)

        out = self.output_head(
            feat.permute(0, 2, 1)
        ).permute(0, 2, 1)  # (B, N, 6)

        delta_xyz = out[:, :, :3]
        normal_raw = out[:, :, 3:]

        xyz_out = xyz + delta_xyz
        xyz_out = torch.clamp(xyz_out, -1.0, 1.0)

        normals_out = F.normalize(
            normal_raw,
            p=2,
            dim=-1,
            eps=1e-8
        )

        return xyz_out, normals_out
