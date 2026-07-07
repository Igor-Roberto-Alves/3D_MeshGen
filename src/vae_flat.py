import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder
from src.decoder_flat import FlatDecoder
from src.Vae import normalize_pc


class VaeFlat(nn.Module):


    def __init__(
        self,
        latent_dim:  int = 512,
        in_channels: int = 6,
        n_latent:    int = 512,
        n_points:    int = 2048,
        seed_dim:    int = 128,
        hidden_dim:  int = 384,
        n_stages:    int = 4,
        fold_dim:    int = 64,
        fold_hidden: int = 256,
        resolution:  int = 16,
        encoder_hidden_dim: int = 64,
        encoder_out_dim:    int = 256,
        encoder_stages:     int = 3,
        encoder_resolution: int = 32,
        decoder_norm:       str = "group", 
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_latent    = n_latent
        self.n_points    = n_points

        self.encoder = GlobalEncoder(
            in_channels, latent_dim,
            hidden_dim=encoder_hidden_dim, out_dim=encoder_out_dim,
            n_stages=encoder_stages, resolution=encoder_resolution,
        )
        self.decoder = FlatDecoder(
            latent_dim=latent_dim, n_latent=n_latent, n_points=n_points, seed_dim=seed_dim,
            hidden_dim=hidden_dim, n_stages=n_stages, fold_dim=fold_dim,
            fold_hidden=fold_hidden, resolution=resolution, norm=decoder_norm,
        )

    def forward(self, x: torch.Tensor):

        x = normalize_pc(x)

        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-10.0, 10.0)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

        xyz_out= self.decoder(z)
        
        return xyz_out, xyz_out, mu, logvar

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
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
