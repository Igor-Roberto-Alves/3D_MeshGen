"""
Pure-PyTorch re-implementation of the building blocks used by the PVD
(Point-Voxel Diffusion) PVCNN2 backbone: SharedMLP, PVConv, PointNetSAModule,
PointNetAModule, PointNetFPModule, Attention, Swish.

The original PVD/PVCNN2 repo implements voxelization / devoxelization and the
set-abstraction grouping with custom CUDA kernels. This module reproduces the
same interfaces with plain torch ops (scatter_add + grid_sample for
voxelize/devoxelize, cdist + topk for ball-query/kNN), following the same
style already used in src/Encoder.py (VoxelBranch/PointBranch), so nothing
here needs a CUDA extension to build on Windows.

Kept in a new file on purpose: PVCNN2Diffusion.py (and this file) are not
imported by any existing training script, so the current VAE / LION-style
latent diffusion pipeline is untouched.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import farthest_point_sample, index_points


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedMLP(nn.Module):
    """Stack of 1x1 (or 1x1x1 via dim=2 -> Conv2d) conv + norm + Swish layers."""

    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        norm = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        layers = []
        c = in_channels
        for oc in out_channels:
            layers += [conv(c, oc, 1, bias=False), norm(oc), Swish()]
            c = oc
        self.layers = nn.Sequential(*layers)
        self.out_channels = out_channels[-1]

    def forward(self, x):
        return self.layers(x)


class SE3d(nn.Module):
    """Squeeze-and-excitation over a voxel grid (B, C, R, R, R)."""

    def __init__(self, channels, reduction=8, relu=True):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True) if relu else Swish(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        s = x.mean(dim=[2, 3, 4])
        s = self.fc(s).view(B, C, 1, 1, 1)
        return x * s


class Attention(nn.Module):
    """Non-local self-attention block. D=1 -> input (B,C,N); D=3 -> input (B,C,R,R,R)."""

    def __init__(self, channels, num_heads=8, D=1):
        super().__init__()
        assert channels % num_heads == 0
        self.D = D
        self.num_heads = num_heads
        conv = nn.Conv1d if D == 1 else nn.Conv3d
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_qkv = conv(channels, channels * 3, 1)
        self.to_out = conv(channels, channels, 1)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        h = self.norm(x)
        qkv = self.to_qkv(h).flatten(2)  # (B, 3C, L)
        L = qkv.shape[-1]
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)  # (B, heads, L, hd)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)
        if self.D != 1:
            out = out.view(B, C, *spatial)
        out = self.to_out(out)
        return x + out


def _voxelize(coords, features, resolution, normalize=True, eps=0.0):
    """coords: (B,3,N), features: (B,C,N) -> voxel grid (B,C,R,R,R), sample_grid (B,3,N) in [-1,1]."""
    B, _, N = coords.shape
    R = resolution
    if normalize:
        center = coords.mean(dim=2, keepdim=True)
        coords = coords - center
        scale = coords.norm(dim=1).max(dim=1, keepdim=True).values
        scale = scale.clamp(min=eps if eps > 0 else 1e-6)
        coords = coords / scale.unsqueeze(1) * 0.49
    norm_coords = (coords + 0.5).clamp(0.0, 1.0 - 1e-6) * R
    idx = norm_coords.long().clamp(0, R - 1)
    flat_idx = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]

    C = features.shape[1]
    voxels = torch.zeros(B, C, R * R * R, device=features.device, dtype=features.dtype)
    count = torch.zeros(B, 1, R * R * R, device=features.device, dtype=features.dtype)
    voxels.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, C, -1), features)
    count.scatter_add_(2, flat_idx.unsqueeze(1), torch.ones(B, 1, N, device=features.device, dtype=features.dtype))
    voxels = (voxels / count.clamp(min=1.0)).view(B, C, R, R, R)

    sample_grid = (norm_coords / R * 2 - 1).clamp(-1.0, 1.0)
    return voxels, sample_grid


def _devoxelize(voxel_feats, sample_grid):
    """voxel_feats: (B,C,R,R,R), sample_grid: (B,3,N) in [-1,1] -> (B,C,N)."""
    N = sample_grid.shape[-1]
    grid = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)  # (B,1,1,N,3)
    grid = grid.expand(-1, 1, 1, N, 3)
    sampled = F.grid_sample(voxel_feats, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return sampled.squeeze(2).squeeze(2)  # (B,C,N)


class PVConv(nn.Module):
    """Point-Voxel convolution: voxel branch (Conv3d) + point branch (SharedMLP), fused."""

    def __init__(self, in_channels, out_channels, kernel_size=3, resolution=32,
                 attention=False, dropout=0.1, with_se=False, with_se_relu=True,
                 normalize=True, eps=0):
        super().__init__()
        self.resolution = resolution
        self.normalize = normalize
        self.eps = eps
        half = max(out_channels // 2, 1)
        pad = kernel_size // 2

        layers = [
            nn.Conv3d(in_channels, half, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(min(8, half), half),
            Swish(),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        layers += [
            nn.Conv3d(half, half, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(min(8, half), half),
            Swish(),
        ]
        self.voxel_layers = nn.Sequential(*layers)
        self.attention = Attention(half, num_heads=8, D=3) if attention else None
        self.se = SE3d(half, relu=with_se_relu) if with_se else None

        self.point_features = SharedMLP(in_channels, out_channels - half, dim=1)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            Swish(),
        )
        self.out_channels = out_channels

    def forward(self, inputs):
        features, coords, temb = inputs
        voxel_feats, sample_grid = _voxelize(coords, features, self.resolution, self.normalize, self.eps)
        voxel_feats = self.voxel_layers(voxel_feats)
        if self.attention is not None:
            voxel_feats = self.attention(voxel_feats)
        if self.se is not None:
            voxel_feats = self.se(voxel_feats)
        v = _devoxelize(voxel_feats, sample_grid)
        p = self.point_features(features)
        out = self.fuse(torch.cat([v, p], dim=1))
        return out, coords, temb


class PointNetSAModule(nn.Module):
    """Set abstraction: FPS + ball-query grouping + SharedMLP + max-pool. Downsamples N -> num_centers."""

    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        self.num_centers = num_centers
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates
        extra = 3 if include_coordinates else 0
        self.mlp = SharedMLP(in_channels + extra, out_channels, dim=2)
        out_channels = out_channels if isinstance(out_channels, (list, tuple)) else [out_channels]
        self.out_channels = out_channels[-1]

    def forward(self, inputs):
        features, coords, temb = inputs
        B, _, N = coords.shape
        xyz = coords.transpose(1, 2).contiguous()  # (B,N,3)

        center_idx = farthest_point_sample(xyz, self.num_centers)  # (B,S)
        centers = index_points(xyz, center_idx)  # (B,S,3)

        dists = torch.cdist(centers, xyz)  # (B,S,N)
        k = min(self.num_neighbors, N)
        knn_dists, knn_idx = dists.topk(k, dim=-1, largest=False)
        out_of_range = knn_dists > self.radius
        nearest = knn_idx[..., :1].expand_as(knn_idx)
        knn_idx = torch.where(out_of_range, nearest, knn_idx)

        grouped_xyz = index_points(xyz, knn_idx.reshape(B, -1)).view(B, self.num_centers, k, 3)
        grouped_xyz = grouped_xyz - centers.unsqueeze(2)

        feat = features.transpose(1, 2).contiguous()  # (B,N,C)
        grouped_feat = index_points(feat, knn_idx.reshape(B, -1)).view(B, self.num_centers, k, -1)
        if self.include_coordinates:
            grouped_feat = torch.cat([grouped_feat, grouped_xyz], dim=-1)
        grouped_feat = grouped_feat.permute(0, 3, 2, 1)  # (B, C, K, S)

        new_feat = self.mlp(grouped_feat).max(dim=2).values  # (B, C', S)
        new_coords = centers.transpose(1, 2).contiguous()  # (B,3,S)
        new_temb = index_points(temb.transpose(1, 2).contiguous(), center_idx).transpose(1, 2).contiguous() \
            if temb is not None else None
        return new_feat, new_coords, new_temb


class PointNetAModule(nn.Module):
    """Global abstraction: SharedMLP + global max-pool over all points -> single "center"."""

    def __init__(self, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        self.include_coordinates = include_coordinates
        extra = 3 if include_coordinates else 0
        self.mlp = SharedMLP(in_channels + extra, out_channels, dim=1)
        out_channels = out_channels if isinstance(out_channels, (list, tuple)) else [out_channels]
        self.out_channels = out_channels[-1]

    def forward(self, inputs):
        features, coords, temb = inputs
        x = torch.cat([features, coords], dim=1) if self.include_coordinates else features
        feat = self.mlp(x)
        pooled = feat.max(dim=2, keepdim=True).values  # (B,C',1)
        new_coords = coords.mean(dim=2, keepdim=True)
        new_temb = temb.mean(dim=2, keepdim=True) if temb is not None else None
        return pooled, new_coords, new_temb


class PointNetFPModule(nn.Module):
    """Feature propagation: 3-NN inverse-distance interpolation from coarse -> fine points, + skip concat + MLP."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels, out_channels, dim=1)
        out_channels = out_channels if isinstance(out_channels, (list, tuple)) else [out_channels]
        self.out_channels = out_channels[-1]

    @staticmethod
    def _interpolate(fine_xyz, coarse_xyz, coarse_values):
        S = coarse_xyz.shape[1]
        if S == 1:
            return coarse_values.expand(-1, -1, fine_xyz.shape[1]).contiguous()
        dists = torch.cdist(fine_xyz, coarse_xyz)  # (B,N,S)
        k = min(3, S)
        knn_dists, knn_idx = dists.topk(k, dim=-1, largest=False)
        weight = 1.0 / knn_dists.clamp(min=1e-8)
        weight = weight / weight.sum(dim=-1, keepdim=True)  # (B,N,k)
        values_t = coarse_values.transpose(1, 2).contiguous()  # (B,S,C)
        gathered = index_points(values_t, knn_idx)  # (B,N,k,C)
        interp = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (B,N,C)
        return interp.transpose(1, 2).contiguous()  # (B,C,N)

    def forward(self, inputs):
        fine_coords, coarse_coords, coarse_feat, skip_feat, temb = inputs
        fine_xyz = fine_coords.transpose(1, 2).contiguous()
        coarse_xyz = coarse_coords.transpose(1, 2).contiguous()

        interp_feat = self._interpolate(fine_xyz, coarse_xyz, coarse_feat)
        interp_temb = self._interpolate(fine_xyz, coarse_xyz, temb) if temb is not None else None

        if skip_feat is not None:
            interp_feat = torch.cat([interp_feat, skip_feat], dim=1)
        new_feat = self.mlp(interp_feat)
        return new_feat, fine_coords, interp_temb
