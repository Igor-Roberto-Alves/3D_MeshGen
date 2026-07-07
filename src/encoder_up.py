import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder, PVConvBlockConditioned  # noqa: F401 (GlobalEncoder re-exported)
from src.utils import farthest_point_sample, index_points


def knn_query(xyz: torch.Tensor, query_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each point in query_xyz, find its k nearest neighbours in xyz.
    xyz:       (B, N, 3)
    query_xyz: (B, M, 3)
    Returns:   (B, M, k) indices into the N axis of xyz.
    """
    diffs = query_xyz.unsqueeze(2) - xyz.unsqueeze(1)   # (B, M, N, 3)
    dists = (diffs ** 2).sum(dim=-1)                    # (B, M, N)
    return dists.topk(k, dim=-1, largest=False).indices  # (B, M, k)


class LocalEncoderUp(nn.Module):
    """
    Local encoder with FPS + k-NN grouping to produce a coarse latent.

    Pipeline
    --------
    Stage 0   : PVConv on all N input points  → (B, N, 64)
    FPS       : N → n_latent anchor positions
    k-NN pool : aggregate N-point features around each anchor → (B, n_latent, 64)
    Stage 1-2 : PVConv on n_latent points     → (B, n_latent, 256)
    fc heads  : mu, logvar                    → (B, n_latent, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 6,
        latent_dim:  int = 3,
        style_dim:   int = 256,
        n_latent:    int = 512,
        k:           int = 16,
    ):
        super().__init__()
        self.n_latent   = n_latent
        self.k          = k
        self.latent_dim = latent_dim

        self.stage0 = PVConvBlockConditioned(in_channels, 64,  style_dim, resolution=32)
        self.stage1 = PVConvBlockConditioned(64,          128, style_dim, resolution=16)
        self.stage2 = PVConvBlockConditioned(128,         256, style_dim, resolution=8)

        self.fc_mu     = nn.Conv1d(256, latent_dim, 1)
        self.fc_logvar = nn.Conv1d(256, latent_dim, 1)
        nn.init.constant_(self.fc_logvar.bias, -6.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        x     : (B, N, in_channels)
        style : (B, style_dim)
        Returns mu, logvar each (B, n_latent, latent_dim)
        """
        xyz  = x[..., :3]
        feat = x

        # Full-resolution feature extraction
        feat = self.stage0(xyz, feat, style)                            # (B, N, 64)

        # FPS: pick n_latent representative points
        fps_idx = farthest_point_sample(xyz, self.n_latent)            # (B, n_latent)
        fps_xyz = index_points(xyz, fps_idx)                           # (B, n_latent, 3)

        # k-NN grouping: for each FPS anchor, pool features from k nearest full-cloud neighbours
        knn_idx = knn_query(xyz, fps_xyz, self.k)                      # (B, n_latent, k)
        B, M, k = knn_idx.shape
        C       = feat.shape[2]
        grouped = index_points(feat, knn_idx.reshape(B, -1))           # (B, n_latent*k, C)
        grouped = grouped.view(B, M, k, C)
        feat    = grouped.max(dim=2).values                            # (B, n_latent, C)

        # Coarse-resolution refinement
        feat = self.stage1(fps_xyz, feat, style)                       # (B, n_latent, 128)
        feat = self.stage2(fps_xyz, feat, style)                       # (B, n_latent, 256)

        f      = feat.permute(0, 2, 1)                                 # (B, 256, n_latent)
        mu     = self.fc_mu(f).permute(0, 2, 1)                       # (B, n_latent, latent_dim)
        logvar = self.fc_logvar(f).permute(0, 2, 1)
        return mu, logvar
