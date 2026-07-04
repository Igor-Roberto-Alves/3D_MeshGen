"""
DecoderPointnet.py
------------------
Two decoder options for the point cloud VAE.

DecoderMLP
    Flat MLP:  z [B, D] → xyz [B, N, 3]
    Simplest, fastest to train, good baseline.
    Tends to produce slightly isotropic reconstructions for complex shapes.

DecoderFolding
    FoldingNet-style (Yang et al., CVPR 2018).
    Folds a 2-D grid into 3-D using two MLP stages, each conditioned on z.
    Crisper on smooth manifold surfaces (aircraft wings, car bodies).
    Output size is ceil(sqrt(N))^2 — always a perfect square.
"""

import math
import torch
import torch.nn as nn


class DecoderMLP(nn.Module):
    """
    z [B, latent_dim]  →  xyz [B, num_points, 3]

    BN + ReLU MLP with increasing hidden dims.
    """
    def __init__(self, latent_dim: int = 256, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),  nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Linear(256,  512),        nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Linear(512,  1024),       nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),       nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.size(0), self.num_points, 3)


class DecoderFolding(nn.Module):
    """
    FoldingNet decoder: z + fixed 2-D grid → xyz [B, M, 3].

    M = ceil(sqrt(num_points))^2  (first perfect square ≥ num_points).
    Two successive folding passes (fold1, fold2), both conditioned on z.
    """
    def __init__(self, latent_dim: int = 256, num_points: int = 2048):
        super().__init__()
        s = math.ceil(math.sqrt(num_points))
        self.num_points = s * s        # actual output count (≥ requested)
        self.grid_size  = s

        # Fixed 2-D grid — not a learnable parameter
        xs   = torch.linspace(-0.5, 0.5, s)
        grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("grid", grid)             # [M, 2]

        # Fold 1: (z concat 2-D) → 3-D
        self.fold1 = nn.Sequential(
            nn.Conv1d(latent_dim + 2, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512,            512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512,              3, 1),
        )
        # Fold 2: (z concat 3-D from fold1) → refined 3-D
        self.fold2 = nn.Sequential(
            nn.Conv1d(latent_dim + 3, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512,            512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512,              3, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, M = z.size(0), self.num_points

        grid = self.grid.T.unsqueeze(0).expand(B, -1, -1)   # [B, 2, M]
        z_r  = z.unsqueeze(-1).expand(-1, -1, M)            # [B, latent_dim, M]

        out1 = self.fold1(torch.cat([z_r, grid], dim=1))    # [B, 3, M]
        out2 = self.fold2(torch.cat([z_r, out1], dim=1))    # [B, 3, M]
        return out2.permute(0, 2, 1)                         # [B, M, 3]
