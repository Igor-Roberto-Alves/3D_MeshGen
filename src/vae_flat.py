import torch
import torch.nn as nn

from src.Encoder import GlobalEncoder
from src.decoder_flat import FlatDecoder
from src.Vae import normalize_pc


class VaeFlat(nn.Module):
    """
    VAE com latente UNICO e flat — sem split global/local, sem "pontos
    latentes" por ancora.

    A forma inteira e resumida por um unico vetor gaussiano
    z ~ N(mu, sigma), mu/logvar de shape (B, latent_dim). O decoder
    (`FlatDecoder`) e StyleGAN-like: um canvas de ancoras constante e
    aprendido, modulado inteiramente por `z` via AdaGN em cada estagio.

    Comparado ao VaeUp (z_g + z_l por ancora), este design elimina:
      - a assimetria KL entre um latente "global" e um "local por ponto"
        (fonte de parte da memorizacao discutida: hubs de ancoras vivas
        vs. mortas);
      - o problema de correspondencia de indice entre amostras diferentes
        (nao ha mais "slot i" por amostra pra interpolar/comparar).
    Em troca, todo o trabalho de diferenciar formas passa por um unico
    vetor — por isso o decoder precisa ser mais potente (ver FlatDecoder).
    """

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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_latent    = n_latent
        self.n_points    = n_points

        self.encoder = GlobalEncoder(in_channels, latent_dim)
        self.decoder = FlatDecoder(
            latent_dim=latent_dim, n_latent=n_latent, n_points=n_points, seed_dim=seed_dim,
            hidden_dim=hidden_dim, n_stages=n_stages, fold_dim=fold_dim,
            fold_hidden=fold_hidden, resolution=resolution,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, in_channels)

        Returns
        -------
        xyz_out    (B, N, 3)
        xyz_coarse (B, n_latent, 3)   — posicoes das ancoras antes do folding
        mu         (B, latent_dim)
        logvar     (B, latent_dim)
        """
        x = normalize_pc(x)

        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-10.0, 10.0)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()

        xyz_out, xyz_coarse = self.decoder(z, return_coarse=True)
        return xyz_out, xyz_coarse, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int = 2048,   # mantido por compat. de API; saida e sempre n_latent x ratio
        device:      torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from the prior N(0,I) and decode."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
