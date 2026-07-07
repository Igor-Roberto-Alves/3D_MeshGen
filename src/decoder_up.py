import math

import torch
import torch.nn as nn

from src.Decoder import PVConvBlockDecoder, SharedMLP


def _make_seeds(ratio: int) -> torch.Tensor:

    side = int(math.ceil(math.sqrt(ratio)))
    t    = torch.linspace(-1.0, 1.0, side)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    grid   = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (side², 2)
    return grid[:ratio]                                          # (ratio, 2)


class LIONDecoderUp(nn.Module):

    def __init__(
        self,
        latent_dim: int = 3,
        style_dim:  int = 256,
        n_latent:   int = 512,
        n_points:   int = 2048,
    ):
        assert n_points % n_latent == 0, (
            f"n_points ({n_points}) must be divisible by n_latent ({n_latent})"
        )
        super().__init__()
        self.n_latent = n_latent
        self.ratio    = n_points // n_latent


        self.pos_head  = nn.Sequential(
            nn.Conv1d(latent_dim, 64, 1), nn.GELU(), nn.Conv1d(64, 3, 1)
        )
        self.feat_proj = SharedMLP([latent_dim, 128, 256])

        self.stage0  = PVConvBlockDecoder(256, 256, style_dim, resolution=16)
        self.stage1  = PVConvBlockDecoder(256, 128, style_dim, resolution=16)
        self.stage2  = PVConvBlockDecoder(128, 64,  style_dim, resolution=16)

        self.refine0 = nn.Conv1d(256, 3, 1)
        self.refine1 = nn.Conv1d(128, 3, 1)

        fold_in = 64 + 2
        self.fold_mlp = nn.Sequential(
            nn.Linear(fold_in, 128), nn.GELU(),
            nn.Linear(128, 64),     nn.GELU(),
            nn.Linear(64, 3),
        )

        seeds = _make_seeds(self.ratio)          
        self.register_buffer("seeds", seeds)

    def forward(
        self,
        z_l: torch.Tensor,
        z_global: torch.Tensor,
        return_coarse: bool = False,
    ):

        B = z_l.shape[0]
        z = z_l.permute(0, 2, 1)                                              # (B, D, n_latent)

        xyz_cur = torch.tanh(self.pos_head(z)).permute(0, 2, 1)              # (B, n_latent, 3)
        feat    = self.feat_proj(z).permute(0, 2, 1)                         # (B, n_latent, 256)

        feat    = self.stage0(xyz_cur, feat, z_global)
        xyz_cur = torch.tanh(
            xyz_cur + self.refine0(feat.permute(0, 2, 1)).permute(0, 2, 1)
        )

        feat    = self.stage1(xyz_cur, feat, z_global)
        xyz_cur = torch.tanh(
            xyz_cur + self.refine1(feat.permute(0, 2, 1)).permute(0, 2, 1)
        )

        feat = self.stage2(xyz_cur, feat, z_global)                          # (B, n_latent, 64)


        feat_rep  = feat.unsqueeze(2).expand(-1, -1, self.ratio, -1)
        seeds_rep = self.seeds.unsqueeze(0).unsqueeze(0).expand(B, self.n_latent, -1, -1)

        fold_in  = torch.cat([feat_rep, seeds_rep], dim=-1)                  # (B, n_latent, ratio, 66)
        delta    = self.fold_mlp(fold_in)                                    # (B, n_latent, ratio, 3)

        xyz_rep  = xyz_cur.unsqueeze(2).expand(-1, -1, self.ratio, -1)       # (B, n_latent, ratio, 3)
        xyz_fine = torch.tanh(xyz_rep + delta)                               # (B, n_latent, ratio, 3)

        xyz_fine = xyz_fine.reshape(B, self.n_latent * self.ratio, 3)         # (B, N, 3)
        if return_coarse:
            return xyz_fine, xyz_cur
        return xyz_fine
