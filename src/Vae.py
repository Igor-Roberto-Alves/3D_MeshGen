import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self, input_dim=6, latent_dim=128
    ):  # input_dim padrão é 6 (XYZ + Normais)
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Usamos Conv1d porque a nuvem é uma lista sequencial de pontos
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )

        # Agrupamento global para aceitar os 2048 pontos e reduzir a dimensionalidade
        self.fc = nn.Linear(512, latent_dim * 2)

        # Decoder reconstrói diretamente a matriz (2048, 6)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048 * 6),  # Gera todos os pontos achatados
        )

    def encode(self, x):
        # x shape esperado: [Batch, 6, 2048]
        h = self.encoder(x)
        # Max pooling ao longo dos pontos (estilo PointNet)
        h = torch.max(h, 2)[0]
        h = self.fc(h)
        mu = h[:, : self.latent_dim]
        logvar = h[:, self.latent_dim :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        # Redimensiona de volta para o formato de imagem/canal esperado pelo treino [Batch, 6, 2048, 1]
        h = h.view(z.size(0), 6, 2048, 1)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
