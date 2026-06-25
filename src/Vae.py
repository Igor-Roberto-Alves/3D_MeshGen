import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Encoder import Encoder, EncoderStyle
from src.Decoder import LIONDecoder


def normalise_xyz(x: torch.Tensor) -> torch.Tensor:
    """
    Normalises the XYZ channels (first 3) of a point cloud to [-1, 1]
    per-sample, leaving normal channels untouched.

    FIX #1: VoxelBranch assumes XYZ in [-1, 1].  Raw point clouds almost
    never satisfy this, causing all points to land in the same voxels
    and the network to learn a unit-cube representation instead of shape.

    x : (B, N, C)  — first 3 channels are XYZ
    returns (B, N, C) with XYZ rescaled to [-1, 1]
    """
    xyz     = x[..., :3]                                        # (B, N, 3)
    rest    = x[..., 3:]                                        # (B, N, C-3)

    xyz_min = xyz.min(dim=1, keepdim=True).values               # (B, 1, 3)
    xyz_max = xyz.max(dim=1, keepdim=True).values

    # Uniform scale: keep aspect ratio, centre at origin
    centre  = (xyz_min + xyz_max) * 0.5                         # (B, 1, 3)
    scale   = (xyz_max - xyz_min).max(dim=-1, keepdim=True).values * 0.5  # (B, 1, 1)
    scale   = scale.clamp(min=1e-6)

    xyz_norm = (xyz - centre) / scale                           # → [-1, 1]
    return torch.cat([xyz_norm, rest], dim=-1)


class Vae(nn.Module):
    def __init__(
        self,
        latent_dim:        int = 256,
        style_dim:         int = 512,
        in_channels:       int = 6,
        num_latent_points: int = 512,
        up_factor:         int = 4,
    ):
        super().__init__()
        self.latent_dim        = latent_dim
        self.style_dim         = style_dim
        self.num_latent_points = num_latent_points

        self.style_encoder = EncoderStyle(in_channels=in_channels, style_dim=style_dim)
        self.encoder       = Encoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
            style_dim=style_dim,
            num_latent_points=num_latent_points,
        )
        self.decoder = LIONDecoder(
            latent_dim=latent_dim,
            style_dim=style_dim,
            input_dim=3,
            up_factor=up_factor,
        )

    # ------------------------------------------------------------------
    def Encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_style, logvar_style = self.style_encoder(x)
        sty = self.style_encoder.z(mu_style, logvar_style)
        logvar_style = torch.clamp(logvar_style, min= -10.0, max = 10.0)

        mu, logvar = self.encoder(x, sty)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar, sty, mu_style, logvar_style

    # ------------------------------------------------------------------
    def Decode(
        self, z: torch.Tensor, style: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z, style)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIX #1: normalise XYZ to [-1, 1] before any PVCNN stage
        x = normalise_xyz(x)

        mu, logvar, sty, mu_style, logvar_style = self.Encode(x)

        # FIX #3: reparametrisation split
        #   - XYZ anchor channels (first 3): use mu directly — no noise,
        #     because logvar there is fixed to -4 → std ≈ 0.135, just enough
        #     to regularise but not destroy anchor positions.
        #   - Feature channels (3:): standard reparametrisation trick.
        std_feat   = torch.exp(0.5 * logvar[:, :, 3:])
        eps_feat   = torch.randn_like(std_feat)

        z_xyz      = mu[:, :, :3]  # Deterministic spatial anchors
        z_feat     = mu[:, :, 3:]  + eps_feat * std_feat  # Stochastic features

        z_points   = torch.cat([z_xyz, z_feat], dim=-1)       # (B, M, 3+latent_dim)

        coords_pred, normals_pred = self.Decode(z_points, sty)
        return coords_pred, normals_pred, mu, logvar, mu_style, logvar_style

    # ------------------------------------------------------------------
    def z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Utility reparametrisation (full tensor, used externally)."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int  = 512,
        device:      torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the prior and decode.

        FIX #5: anchors are sampled from N(0, I) — matching what the KL
        term regularises the encoder towards — rather than a hand-crafted
        sphere distribution that mismatches the trained prior.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        sty_sample  = torch.randn(num_samples, self.style_dim,   device=device)
        xyz_anchors = torch.randn(num_samples, num_points, 3,    device=device)
        feat_sample = torch.randn(num_samples, num_points, self.latent_dim, device=device)

        # Clamp anchors to [-1, 1] so VoxelBranch sees valid coordinates
        xyz_anchors = torch.clamp(xyz_anchors, -1.0, 1.0)

        z_points = torch.cat([xyz_anchors, feat_sample], dim=-1)
        coords_pred, normals_pred = self.Decode(z_points, sty_sample)
        return coords_pred, normals_pred