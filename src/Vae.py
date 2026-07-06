import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder, LocalEncoder
from src.Decoder import LIONDecoder


def normalize_pc(x: torch.Tensor) -> torch.Tensor:

    xyz  = x[..., :3]
    rest = x[..., 3:]
    centre = xyz.mean(dim=1, keepdim=True)                                 # (B, 1, 3)
    xyz_c  = xyz - centre

    scale  = (xyz_c.abs()
               .max(dim=-1, keepdim=True).values   # per-point max over xyz
               .max(dim=1,  keepdim=True).values    # global max over points
               .clamp(min=1e-6))                    # (B, 1, 1)
    return torch.cat([xyz_c / scale, rest], dim=-1)


class Vae(nn.Module):


    def __init__(
        self,
        latent_dim:  int = 3,
        style_dim:   int = 256,
        in_channels: int = 6,
    ):
        super().__init__()
        self.style_dim   = style_dim
        self.latent_dim  = latent_dim

        self.global_encoder = GlobalEncoder(in_channels, style_dim)
        self.local_encoder  = LocalEncoder(in_channels, latent_dim, style_dim)
        self.decoder        = LIONDecoder(latent_dim, style_dim)


    def forward(self, x: torch.Tensor):
 
        x = normalize_pc(x)

        # --- Global level ---
        mu_g, logvar_g = self.global_encoder(x)
        logvar_g = logvar_g.clamp(-10.0, 10.0)
        z_g = mu_g + torch.randn_like(mu_g) * (0.5 * logvar_g).exp()

        # --- Local lev ---
        mu_l, logvar_l = self.local_encoder(x, z_g)
        logvar_l = logvar_l.clamp(-10.0, 10.0)
        z_l = mu_l + torch.randn_like(mu_l) * (0.5 * logvar_l).exp()

        xyz_out = self.decoder(z_l, z_g)
        return xyz_out, mu_l, logvar_l, mu_g, logvar_g


    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int = 2048,
        device:      torch.device | None = None,
    ) -> torch.Tensor:

        if device is None:
            device = next(self.parameters()).device
        self.eval()

        z_g = torch.randn(num_samples, self.style_dim,              device=device)
        z_l = torch.randn(num_samples, num_points, self.latent_dim, device=device)
        return self.decoder(z_l, z_g)
