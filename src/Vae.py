import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder, ShapeEncoder
from src.Decoder import FlatDecoder


def normalize_pc(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise xyz to roughly [-1, 1] per sample (uniform scale, centred).
    Normal channels (if present) are left untouched.
    x: (B, N, C) where C >= 3
    """
    xyz  = x[..., :3]
    rest = x[..., 3:]
    centre = xyz.mean(dim=1, keepdim=True)
    xyz_c  = xyz - centre
    scale  = (xyz_c.abs()
               .max(dim=-1, keepdim=True).values
               .max(dim=1,  keepdim=True).values
               .clamp(min=1e-6))
    return torch.cat([xyz_c / scale, rest], dim=-1)


class Vae(nn.Module):
    """
    Flat two-vector VAE for 3D point cloud generation.

    Two independent latent vectors
    --------------------------------
    z_g ∈ ℝ^{style_dim}   — global style, from GlobalEncoder (2 SA blocks)
    z_l ∈ ℝ^{latent_size} — flat shape code, from ShapeEncoder (4 SA blocks → pool)

    The decoder generates the full point cloud from (z_g, z_l) with NO coordinate
    shortcuts — both vectors must carry real geometric information for the decoder
    to reconstruct the shape, eliminating the posterior collapse of z_g seen in
    LION's per-point latent formulation.
    """

    def __init__(
        self,
        latent_size: int = 1024,
        style_dim:   int = 128,
        in_channels: int = 6,
        num_points:  int = 2048,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.style_dim   = style_dim
        self.num_points  = num_points

        self.style_encoder = GlobalEncoder(in_channels, style_dim)
        self.shape_encoder = ShapeEncoder(in_channels, latent_size)
        self.decoder       = FlatDecoder(latent_size, style_dim, num_points)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, in_channels)

        Returns
        -------
        xyz_out  (B, N, 3)
        mu_l     (B, latent_size)
        logvar_l (B, latent_size)
        mu_g     (B, style_dim)
        logvar_g (B, style_dim)
        """
        x = normalize_pc(x)

        mu_g, logvar_g = self.style_encoder(x)
        logvar_g = logvar_g.clamp(-10.0, 10.0)
        z_g = mu_g + torch.randn_like(mu_g) * (0.5 * logvar_g).exp()

        mu_l, logvar_l = self.shape_encoder(x)
        logvar_l = logvar_l.clamp(-10.0, 10.0)
        z_l = mu_l + torch.randn_like(mu_l) * (0.5 * logvar_l).exp()

        xyz_out = self.decoder(z_l, z_g)
        return xyz_out, mu_l, logvar_l, mu_g, logvar_g

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using posterior means (no reparameterisation noise)."""
        self.eval()
        x = normalize_pc(x)
        mu_g, _ = self.style_encoder(x)
        mu_l, _ = self.shape_encoder(x)
        return self.decoder(mu_l, mu_g)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from the prior N(0, I) and decode."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z_g = torch.randn(num_samples, self.style_dim,   device=device)
        z_l = torch.randn(num_samples, self.latent_size, device=device)
        return self.decoder(z_l, z_g)
