import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder, LocalEncoder
from src.Decoder import LIONDecoder


def normalize_pc(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise xyz to roughly [-1, 1] per sample (uniform scale, centred).
    Normal channels (if present) are left untouched.
    x: (B, N, C) where C >= 3
    """
    xyz  = x[..., :3]
    rest = x[..., 3:]
    centre = xyz.mean(dim=1, keepdim=True)                                 # (B, 1, 3)
    xyz_c  = xyz - centre
    # Largest absolute coordinate value → scale so max coord ≈ 1
    scale  = (xyz_c.abs()
               .max(dim=-1, keepdim=True).values   # per-point max over xyz
               .max(dim=1,  keepdim=True).values    # global max over points
               .clamp(min=1e-6))                    # (B, 1, 1)
    return torch.cat([xyz_c / scale, rest], dim=-1)


class Vae(nn.Module):
    """
    Hierarchical Point-Cloud VAE following the LION paper (NeurIPS 2022).

    Two-level latent space
    ----------------------
    z_g  (B, style_dim)         – global shape prior, N(0,I)
    z_l  (B, N, latent_dim)     – local per-point prior, N(0,I)

    Latent points passed to the decoder:
        z_local = cat(xyz_anchors, z_l)   (B, N, 3 + latent_dim)

    The decoder refines anchor positions under global context z_g.
    """

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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x: (B, N, 6)  — raw point cloud [xyz | normals]

        Returns
        -------
        xyz_out   (B, N, 3)           reconstructed positions (normalised scale)
        mu_l      (B, N, latent_dim)  local encoder mean
        logvar_l  (B, N, latent_dim)  local encoder log-variance
        mu_g      (B, style_dim)      global encoder mean
        logvar_g  (B, style_dim)      global encoder log-variance
        """
        x = normalize_pc(x)

        # --- Global level ---
        mu_g, logvar_g = self.global_encoder(x)
        logvar_g = logvar_g.clamp(-10.0, 10.0)
        z_g = mu_g + torch.randn_like(mu_g) * (0.5 * logvar_g).exp()

        # --- Local level (conditioned on z_g) ---
        mu_l, logvar_l = self.local_encoder(x, z_g)
        logvar_l = logvar_l.clamp(-10.0, 10.0)
        z_l = mu_l + torch.randn_like(mu_l) * (0.5 * logvar_l).exp()

        # Anchor positions are the normalised input xyz (no noise added)
        xyz_anchor = x[..., :3]
        z_local = torch.cat([xyz_anchor, z_l], dim=-1)

        xyz_out = self.decoder(z_local, z_g)
        return xyz_out, mu_l, logvar_l, mu_g, logvar_g

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int = 2048,
        device:      torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from the prior N(0,I) and decode."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        z_g         = torch.randn(num_samples, self.style_dim,              device=device)
        xyz_anchors = torch.randn(num_samples, num_points, 3,               device=device).clamp(-1.0, 1.0)
        z_l         = torch.randn(num_samples, num_points, self.latent_dim, device=device)

        z_local = torch.cat([xyz_anchors, z_l], dim=-1)
        return self.decoder(z_local, z_g)
