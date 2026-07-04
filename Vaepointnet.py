"""
Vaepointnet.py
--------------
VAE combining EncoderPointnet with DecoderMLP or DecoderFolding.

forward() returns (recon, mu, logvar, A_feat) so the training loop can
compute Chamfer + KL and the optional T-net orthogonality regularisation.
"""

import torch
import torch.nn as nn

from EncoderPointnet import EncoderPointnet, tnet_regularization   # noqa: F401 (re-exported)
from DecoderPointnet  import DecoderMLP, DecoderFolding            # noqa: F401


class VaePointnet(nn.Module):
    def __init__(self, encoder: EncoderPointnet, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # ------------------------------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar.clamp(-10.0, 10.0)).exp()
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : [B, in_dim, N]

        Returns
        -------
        recon   [B, N_out, 3]      — reconstructed xyz
        mu      [B, latent_dim]
        logvar  [B, latent_dim]
        A_feat  [B, 64, 64]        — feature T-net matrix for regularisation
        """
        mu, logvar, A_feat = self.encoder(x)
        z     = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, A_feat

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, device) -> torch.Tensor:
        """Sample n shapes from the prior p(z) = N(0, I)."""
        latent_dim = self.encoder.fc_mu.out_features
        z = torch.randn(n, latent_dim, device=device)
        return self.decoder(z)                              # [n, N_out, 3]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction via mu (no sampling noise)."""
        mu, _, _ = self.encoder(x)
        return self.decoder(mu)
