import torch
import torch.nn as nn
from typing import Tuple



def farthest_point_sampling(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Iterative FPS on a batch of point clouds.
    xyz : (B, N, C)  — only the first 3 channels are used for distance.
    returns idx : (B, n_samples)  long indices into N
    """
    B, N, _ = xyz.shape
    device   = xyz.device

    idx      = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist     = torch.full((B, N), 1e10, device=device)
    centroid = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        idx[:, i] = centroid
        c    = xyz[torch.arange(B, device=device), centroid, :3].unsqueeze(1)  # (B,1,3)
        d    = ((xyz[..., :3] - c) ** 2).sum(-1)                               # (B,N)
        dist = torch.min(dist, d)
        centroid = dist.argmax(dim=-1)

    return idx



def knn_group(
    xyz: torch.Tensor, centers: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xyz     : (B, N, C)
    centers : (B, M, C)
    returns:
        grouped : (B, M, k, C)  — relative XYZ + absolute normals
        idx     : (B, M, k)
    """
    B, _, C = xyz.shape
    M       = centers.shape[1]

    # Spatial-only pairwise distances
    diff    = centers[..., :3].unsqueeze(2) - xyz[..., :3].unsqueeze(1)  # (B,M,N,3)
    sq_dist = (diff ** 2).sum(-1)                                         # (B,M,N)
    idx     = sq_dist.topk(k, largest=False, dim=-1).indices              # (B,M,k)

    # Gather all C channels for the k neighbours
    idx_flat = idx.reshape(B, -1)                                         # (B, M*k)
    pts      = torch.gather(xyz, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
    grouped  = pts.reshape(B, M, k, C)                                    # (B,M,k,C)

    # Relative XYZ only — normals are left absolute
    grouped = grouped.clone()
    grouped[..., :3] -= centers[..., :3].unsqueeze(2)

    return grouped, idx


class PatchEmbed(nn.Module):
    """
    Dual-stream mini-PointNet.
    Input : (B, M, k, 6)  — [rel_xyz | normals]
    Output: (B, M, out_ch)
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 256, k: int = 32):
        super().__init__()
        assert in_ch == 6, "PatchEmbed expects 6-channel input (xyz + normals)"
        half = out_ch // 2

        # Stream A: local geometry (relative XYZ)
        self.geo_net = nn.Sequential(
            nn.Linear(3, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
            nn.Linear(half, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
        )

        # Stream B: surface orientation (normals)
        self.norm_net = nn.Sequential(
            nn.Linear(3, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
            nn.Linear(half, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
        )

        # Fusion: combine both streams before max-pool
        self.fuse = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.GELU(),
        )

        self.k      = k
        self.out_ch = out_ch

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        """grouped : (B, M, k, 6)"""
        B, M, k, _ = grouped.shape
        N           = B * M * k

        xyz_rel  = grouped[..., :3].reshape(N, 3)
        normals  = grouped[..., 3:].reshape(N, 3)

        geo_feat  = self.geo_net(xyz_rel)   # (N, half)
        norm_feat = self.norm_net(normals)  # (N, half)

        fused = self.fuse(torch.cat([geo_feat, norm_feat], dim=-1))  # (N, out_ch)
        fused = fused.reshape(B * M, k, self.out_ch)

        token = fused.max(dim=1).values     # max-pool over k neighbours
        return token.reshape(B, M, self.out_ch)



class PositionalEncoding(nn.Module):
    """
    MLP positional encoding from 3-D spatial coordinates only.
    Input : (B, M, 3)  — XYZ of patch centres
    Output: (B, M, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz : (..., 3)"""
        return self.mlp(xyz)