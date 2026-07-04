"""
EncoderPointnet.py
------------------
PointNet encoder for point cloud VAE.

Input:  x [B, in_dim, N]   — in_dim=3 (xyz) or 6 (xyz + normals)
Output: mu [B, latent_dim], logvar [B, latent_dim], A_feat [B, 64, 64]

A_feat is the feature T-net matrix used for optional orthogonality
regularisation (tnet_regularization). Pass it to the loss in train_vaepointnet.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Tnet(nn.Module):
    """Mini-PointNet that predicts a dim×dim alignment matrix."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = nn.Sequential(
            nn.Conv1d(dim, 64,   1), nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64,  128,  1), nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, dim * dim),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.conv(x).max(dim=-1)[0]                      # [B, 1024]
        mat = self.fc(h).view(B, self.dim, self.dim)
        return mat + torch.eye(self.dim, device=x.device, dtype=x.dtype)


def tnet_regularization(A: torch.Tensor) -> torch.Tensor:
    """||A @ Aᵀ − I||²_F  — penalises the feature T-net leaving orthogonality."""
    d = A.size(1)
    I = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0)
    diff = torch.bmm(A, A.transpose(1, 2)) - I
    return diff.pow(2).sum(dim=(1, 2)).mean()


class EncoderPointnet(nn.Module):
    """
    PointNet encoder → Gaussian latent (mu, logvar).

    Parameters
    ----------
    latent_dim  : size of z
    in_dim      : 3 (xyz only) or 6 (xyz + normals)
    global_dim  : width of the global feature vector before projection (default 1024)
    """
    def __init__(self, latent_dim: int = 256, in_dim: int = 3, global_dim: int = 1024):
        super().__init__()
        self.in_dim = in_dim

        # Input T-net always operates on xyz only for geometric interpretability
        self.tnet_in = _Tnet(3)

        # Shared MLP 1: project to 64-D (handles both in_dim=3 and in_dim=6)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64,     64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )

        # Feature T-net on 64-D feature space
        self.tnet_feat = _Tnet(64)

        # Shared MLP 2: 64 → global_dim
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64,  64,         1), nn.BatchNorm1d(64),         nn.ReLU(inplace=True),
            nn.Conv1d(64,  128,        1), nn.BatchNorm1d(128),        nn.ReLU(inplace=True),
            nn.Conv1d(128, global_dim, 1), nn.BatchNorm1d(global_dim), nn.ReLU(inplace=True),
        )

        # Projection to 256-D before mu/logvar heads
        self.proj = nn.Sequential(
            nn.Linear(global_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512,        256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )

        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        x : [B, in_dim, N]
        Returns
        -------
        mu     [B, latent_dim]
        logvar [B, latent_dim]
        A_feat [B, 64, 64]   — feature T-net matrix for optional regularisation
        """
        # Align xyz with the input T-net; rotate normals by the same matrix
        T_in = self.tnet_in(x[:, :3])               # [B, 3, 3]
        xyz  = torch.bmm(T_in, x[:, :3])            # [B, 3, N]
        if self.in_dim == 6:
            nrm = torch.bmm(T_in, x[:, 3:])         # [B, 3, N]
            x   = torch.cat([xyz, nrm], dim=1)       # [B, 6, N]
        else:
            x = xyz                                   # [B, 3, N]

        x = self.mlp1(x)                             # [B, 64, N]

        A_feat = self.tnet_feat(x)                   # [B, 64, 64]
        x      = torch.bmm(A_feat, x)               # [B, 64, N]

        x = self.mlp2(x)                             # [B, global_dim, N]
        x = x.max(dim=-1)[0]                         # [B, global_dim]

        h      = self.proj(x)                        # [B, 256]
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, A_feat
