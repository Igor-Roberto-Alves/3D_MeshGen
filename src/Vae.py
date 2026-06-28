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
    centre = xyz.mean(dim=1, keepdim=True)
    xyz_c  = xyz - centre
    scale  = (xyz_c.abs()
               .max(dim=-1, keepdim=True).values
               .max(dim=1,  keepdim=True).values
               .clamp(min=1e-6))
    return torch.cat([xyz_c / scale, rest], dim=-1)


class Vae(nn.Module):
    """
    Hierarchical Point-Cloud VAE following the LION paper (NeurIPS 2022).

    Two-level latent space
    ----------------------
    z_g  (B, style_dim)              – global shape prior
    z_l  (B, N, 3 + latent_dim)     – local per-point latent passed to decoder:
                                         z_l[:,:,:3] = xyz + 0.01 * delta_pos
                                         z_l[:,:,3:] = delta_feat  (+ noise)

    The encoder outputs RAW deltas (mu_l ≈ 0 at init). The position skip
    is applied in forward() so KL is computed on the small delta, not on xyz.
    This follows LION paper D.1: "z_l = xyz + 0.01 * mu_l".
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
        self.total_z_dim = 3 + latent_dim   # full channels of z_l

        self.global_encoder = GlobalEncoder(in_channels, style_dim)
        # LocalEncoder outputs (3 + latent_dim) channels with position anchor.
        self.local_encoder  = LocalEncoder(in_channels, latent_dim, style_dim)
        # Decoder expects z_l with (3 + latent_dim) channels.
        self.decoder        = LIONDecoder(latent_dim, style_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, pos_noise_std: float = 0.0):
        """
        x: (B, N, 6)  — raw point cloud [xyz | normals]
        pos_noise_std: std of Gaussian noise added to z_l position channels
                       before decoding.  Breaks the position shortcut and forces
                       the decoder to rely on z_g for spatial correction.
                       Set to 0 (default) to disable; typical values: 0.02–0.10.

        Returns
        -------
        xyz_out   (B, N, 3)            reconstructed positions
        nrm_out   (B, N, 3)            predicted normals
        mu_l      (B, N, 3+latent_dim) local encoder mean (includes position anchor)
        logvar_l  (B, N, 3+latent_dim) local encoder log-variance
        mu_g      (B, style_dim)       global encoder mean
        logvar_g  (B, style_dim)       global encoder log-variance
        """
        x = normalize_pc(x)

        # --- Global level ---
        mu_g, logvar_g = self.global_encoder(x)
        logvar_g = logvar_g.clamp(-10.0, 10.0)
        z_g = mu_g + torch.randn_like(mu_g) * (0.5 * logvar_g).exp()

        # --- Local level ---
        # Encoder returns raw deltas (mu ≈ 0 at init, logvar ≈ -6).
        # KL is computed on these raw deltas — NOT on xyz.
        mu_l, logvar_l = self.local_encoder(x, z_g)
        logvar_l = logvar_l.clamp(-10.0, 10.0)
        eps_l    = torch.randn_like(mu_l) * (0.5 * logvar_l).exp()
        delta    = mu_l + eps_l                          # (B, N, total_z_dim)

        # LION paper D.1 skip: position channels use xyz + 0.01 * delta.
        # Feature channels (3:) are passed as-is (no position constraint).
        xyz_anchor = x[..., :3]
        z_l = delta.clone()
        z_l[..., :3] = xyz_anchor + 0.01 * delta[..., :3]

        # Optional position noise: breaks shortcut, forces decoder to use z_g.
        if pos_noise_std > 0.0 and self.training:
            z_l = z_l.clone()
            z_l[..., :3] = z_l[..., :3] + torch.randn_like(z_l[..., :3]) * pos_noise_std

        xyz_out, nrm_out = self.decoder(z_l, z_g)
        return xyz_out, nrm_out, mu_l, logvar_l, mu_g, logvar_g

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor):
        """
        Deterministic reconstruction using posterior means — no reparameterization noise.
        Use this for visualisation; training uses the sampled forward().
        """
        self.eval()
        x = normalize_pc(x)

        mu_g, _  = self.global_encoder(x)
        mu_l, _  = self.local_encoder(x, mu_g)   # raw deltas (≈0 at init)
        xyz_anchor = x[..., :3]
        z_l        = mu_l.clone()
        z_l[..., :3] = xyz_anchor + 0.01 * mu_l[..., :3]
        xyz_out, nrm_out = self.decoder(z_l, mu_g)
        return xyz_out, nrm_out

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

        z_g = torch.randn(num_samples, self.style_dim, device=device)
        z_l = torch.randn(num_samples, num_points, self.total_z_dim, device=device)
        xyz_out, nrm_out = self.decoder(z_l, z_g)
        return xyz_out, nrm_out
