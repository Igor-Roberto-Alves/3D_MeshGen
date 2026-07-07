import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder
from src.encoder_up import LocalEncoderUp
from src.decoder_up import LIONDecoderUp
from src.Vae import normalize_pc


class VaeUp(nn.Module):
    """
    Hierarchical Point-Cloud VAE with FPS-compressed local latent.

    Two-level latent space
    ----------------------
    z_g  (B, style_dim)               — global shape prior,  N(0,I)
    z_l  (B, n_latent, latent_dim)    — local coarse prior,  N(0,I)

    The encoder compresses N=2048 points → n_latent via FPS + k-NN pooling.
    The decoder expands n_latent → N via folding (n_latent × ratio = N).
    """

    def __init__(
        self,
        latent_dim:     int   = 3,
        style_dim:      int   = 256,
        in_channels:    int   = 6,
        n_latent:       int   = 512,
        n_points:       int   = 2048,
        k:              int   = 16,
        zg_dropout_p:   float = 0.0,
    ):
        super().__init__()
        self.style_dim     = style_dim
        self.latent_dim    = latent_dim
        self.n_latent      = n_latent
        self.n_points      = n_points
        self.zg_dropout_p  = zg_dropout_p

        self.global_encoder = GlobalEncoder(in_channels, style_dim)
        self.local_encoder  = LocalEncoderUp(in_channels, latent_dim, style_dim, n_latent, k)
        self.decoder        = LIONDecoderUp(latent_dim, style_dim, n_latent, n_points)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, in_channels)

        Returns
        -------
        xyz_out    (B, N, 3)
        xyz_coarse (B, n_latent, 3)   — coarse 512-point output before folding
        mu_l       (B, n_latent, latent_dim)
        logvar_l   (B, n_latent, latent_dim)
        mu_g       (B, style_dim)
        logvar_g   (B, style_dim)
        """
        x = normalize_pc(x)

        mu_g, logvar_g = self.global_encoder(x)
        logvar_g = logvar_g.clamp(-10.0, 10.0)
        z_g = mu_g + torch.randn_like(mu_g) * (0.5 * logvar_g).exp()

        mu_l, logvar_l = self.local_encoder(x, z_g)
        logvar_l = logvar_l.clamp(-10.0, 10.0)
        z_l = mu_l + torch.randn_like(mu_l) * (0.5 * logvar_l).exp()

        # Randomly zero z_g seen by the decoder so z_l must carry structure.
        # The encoder always sees the full z_g; only the decoder is blinded.
        if self.training and self.zg_dropout_p > 0.0:
            mask = (torch.rand(z_g.shape[0], 1, device=z_g.device) > self.zg_dropout_p).float()
            z_g_dec = z_g * mask
        else:
            z_g_dec = z_g

        xyz_out, xyz_coarse = self.decoder(z_l, z_g_dec, return_coarse=True)
        return xyz_out, xyz_coarse, mu_l, logvar_l, mu_g, logvar_g

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int = 2048,   # kept for API compat; output is always n_latent × ratio
        device:      torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from the prior N(0,I) and decode."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z_g = torch.randn(num_samples, self.style_dim,                device=device)
        z_l = torch.randn(num_samples, self.n_latent, self.latent_dim, device=device)
        return self.decoder(z_l, z_g)
