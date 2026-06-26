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
# LION Decoder  —  incremental xyz refinement
# ---------------------------------------------------------------------------

class LIONDecoder(nn.Module):
    """
    Point-cloud decoder conditioned on global shape latent z_g.

    Input:  z_l      (B, N, latent_dim)  — purely stochastic per-point latent
            z_global (B, style_dim)      — global shape context
    Output: xyz_out  (B, N, 3)          — decoded point positions

    No anchor bypass: positions are decoded entirely from the stochastic
    latent so the VAE latent space is a proper generative prior.

    Incremental refinement
    ----------------------
    Instead of predicting absolute positions in a single shot, the decoder
    progressively corrects an xyz estimate after each PVCNN stage:

        xyz_0  = tanh( pos_head(z_l) )            # coarse (Level 0)
        xyz_1  = tanh( xyz_0 + Δ0(feat_stage0) )  # refined (Level 1)
        xyz_2  = tanh( xyz_1 + Δ1(feat_stage1) )  # refined (Level 2)
        output = tanh( xyz_2 + Δ2(feat_stage2) )  # final   (Level 3)

    Benefits:
    • Each PVCNN stage receives spatial coordinates that match its own
      feature scale, giving the voxelisation meaningful content from the start.
    • pos_head gets a direct gradient path to the final loss through the
      residual chain — preventing weight-decay from killing the spread of
      xyz_0 and collapsing all points to the centre voxel.
    • All residual heads are zero-initialised, so training begins with the
      network outputting xyz_0 and gradually learning the corrections.
    """
    def __init__(self, latent_dim=3, style_dim=256):
        super().__init__()

        # ---- Coarse position head ----------------------------------------
        # Maps z_l → initial xyz estimate used for voxelisation.
        # The last layer is initialised with std=1.0 (vs Kaiming ~0.18) so
        # pre-tanh values are already large at epoch 0, keeping points spread
        # across the voxel grid and preventing weight-decay from shrinking
        # them back to zero.
        self.pos_head = nn.Sequential(
            nn.Conv1d(latent_dim, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1),
        )
        nn.init.normal_(self.pos_head[-1].weight, std=1.0)
        nn.init.zeros_(self.pos_head[-1].bias)

        self.feat_proj = SharedMLP([latent_dim, 128, 256])

        self.stage0 = PVConvBlockDecoder(256, 256, style_dim, resolution=16)
        self.stage1 = PVConvBlockDecoder(256, 128, style_dim, resolution=32)
        self.stage2 = PVConvBlockDecoder(128, 64,  style_dim, resolution=32)

        # ---- Incremental refinement heads --------------------------------
        # One lightweight Conv1d per stage: predicts Δxyz from stage features.
        # Zero-init on both weight and bias → Δxyz = 0 at epoch 0, so the
        # network starts from xyz_0 and learns corrections from scratch.
        self.refine0 = nn.Conv1d(256, 3, 1)   # after stage0 (256-ch features)
        self.refine1 = nn.Conv1d(128, 3, 1)   # after stage1 (128-ch features)
        nn.init.zeros_(self.refine0.weight);  nn.init.zeros_(self.refine0.bias)
        nn.init.zeros_(self.refine1.weight);  nn.init.zeros_(self.refine1.bias)

        # ---- Final residual head -----------------------------------------
        # Predicts the last Δxyz on top of the twice-refined xyz_2.
        # Zero-init on the last linear layer (same reason as refine heads).
        self.output_head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, z_l, z_global):
        # z_l:      (B, N, latent_dim)
        # z_global: (B, style_dim)
        z = z_l.permute(0, 2, 1)  # (B, latent_dim, N)

        # ------------------------------------------------------------------
        # Level 0 — coarse xyz from the latent (spread across voxel grid)
        # ------------------------------------------------------------------
        xyz = torch.tanh(self.pos_head(z)).permute(0, 2, 1)  # (B, N, 3) ∈ (-1,1)

        feat = self.feat_proj(z).permute(0, 2, 1)             # (B, N, 256)

        # ------------------------------------------------------------------
        # Level 1 — stage0 sees the coarse xyz, then produces a first delta
        # ------------------------------------------------------------------
        feat = self.stage0(xyz, feat, z_global)               # (B, N, 256)
        # Δ0: per-point 3D correction predicted from stage0 features
        xyz  = torch.tanh(xyz + self.refine0(feat.permute(0, 2, 1)).permute(0, 2, 1))

        # ------------------------------------------------------------------
        # Level 2 — stage1 sees the level-1 xyz, then refines again
        # ------------------------------------------------------------------
        feat = self.stage1(xyz, feat, z_global)               # (B, N, 128)
        # Δ1: second correction — voxelisation now uses a better spatial grid
        xyz  = torch.tanh(xyz + self.refine1(feat.permute(0, 2, 1)).permute(0, 2, 1))

        # ------------------------------------------------------------------
        # Level 3 — stage2 uses the twice-refined xyz for fine detail
        # ------------------------------------------------------------------
        feat  = self.stage2(xyz, feat, z_global)              # (B, N, 64)
        # Δ2: final small correction; tanh keeps output bounded in (-1, 1)
        delta = self.output_head(feat.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.tanh(xyz + delta)
