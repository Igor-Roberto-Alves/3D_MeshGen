import math

import torch
import torch.nn as nn

from src.Decoder import PVConvBlockDecoder, SharedMLP
from src.utils import channel_schedule


def _make_seeds(ratio: int) -> torch.Tensor:
    side = int(math.ceil(math.sqrt(ratio)))
    t    = torch.linspace(-1.0, 1.0, side)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    grid   = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  #
    return grid[:ratio]                                          


class FlatDecoder(nn.Module):

    def __init__(
        self,
        latent_dim:  int = 512,
        n_latent:    int = 512,
        n_points:    int = 2048,
        seed_dim:    int = 128,
        hidden_dim:  int = 384,
        n_stages:    int = 4,
        fold_dim:    int = 64,
        fold_hidden: int = 256,
        resolution:  int = 16,
        norm:        str = "group",   
    ):
        assert n_points % n_latent == 0, (
            f"n_points ({n_points}) must be divisible by n_latent ({n_latent})"
        )
        assert n_stages >= 1, "n_stages must be >= 1"
        super().__init__()
        self.n_latent = n_latent
        self.ratio    = n_points // n_latent
        self.seed_dim = seed_dim


        channels = channel_schedule(hidden_dim, fold_dim, n_stages)
        hidden_dim_actual = channels[0]
        fold_dim_actual   = channels[-1]

    
        self.anchor_embed = nn.Parameter(torch.randn(n_latent, seed_dim) * 0.02)

        self.pos_head  = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1), nn.GELU(), nn.Conv1d(64, 3, 1)
        )
        self.feat_proj = SharedMLP([seed_dim, hidden_dim_actual, hidden_dim_actual], norm=norm)
        self.stages = nn.ModuleList([
            PVConvBlockDecoder(channels[i], channels[i + 1], latent_dim,
                               resolution=resolution, norm=norm)
            for i in range(n_stages)
        ])
 
        self.refines = nn.ModuleList([
            nn.Conv1d(channels[i + 1], 3, 1) for i in range(n_stages - 1)
        ])

        fold_in = fold_dim_actual + 2
        self.fold_mlp = nn.Sequential(
            nn.Linear(fold_in, fold_hidden),              nn.GELU(),
            nn.Linear(fold_hidden, fold_hidden // 2),      nn.GELU(),
            nn.Linear(fold_hidden // 2, fold_dim_actual),  nn.GELU(),
            nn.Linear(fold_dim_actual, 3),
        )

        seeds = _make_seeds(self.ratio)          
        self.register_buffer("seeds", seeds)

    def forward(self, z: torch.Tensor, return_coarse: bool = False):

        B = z.shape[0]

        anchor   = self.anchor_embed.unsqueeze(0).expand(B, -1, -1)   # (B, n_latent, seed_dim)
        anchor_c = anchor.permute(0, 2, 1)                             # (B, seed_dim, n_latent)

  
        xyz_cur = torch.tanh(self.pos_head(anchor_c)).permute(0, 2, 1)  # (B, n_latent, 3)
        feat    = self.feat_proj(anchor_c).permute(0, 2, 1)             # (B, n_latent, hidden_dim)

        for i, stage in enumerate(self.stages):
            feat = stage(xyz_cur, feat, z)
            if i < len(self.refines):
                xyz_cur = torch.tanh(
                    xyz_cur + self.refines[i](feat.permute(0, 2, 1)).permute(0, 2, 1)
                )


        feat_rep  = feat.unsqueeze(2).expand(-1, -1, self.ratio, -1)
        seeds_rep = self.seeds.unsqueeze(0).unsqueeze(0).expand(B, self.n_latent, -1, -1)

        fold_in  = torch.cat([feat_rep, seeds_rep], dim=-1)
        delta    = self.fold_mlp(fold_in)

        xyz_rep  = xyz_cur.unsqueeze(2).expand(-1, -1, self.ratio, -1)
        xyz_fine = torch.tanh(xyz_rep + delta)

        xyz_fine = xyz_fine.reshape(B, self.n_latent * self.ratio, 3)
    
        return xyz_fine
