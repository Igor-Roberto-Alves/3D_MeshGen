import torch
import torch.nn as nn


class AdaGNBlock(nn.Module):
    """Injeta o vetor latente Z no estilo das características dos pontos"""

    def __init__(self, channels, latent_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.fc = nn.Linear(latent_dim, channels * 2)  # Saca escala e bias

    def forward(self, x, z):

        normED = self.norm(x)

        style = self.fc(z).unsqueeze(-1)
        gamma, beta = torch.chunk(style, 2, dim=1)

        return normED * (1 + gamma) + beta


class MiniLionDecoder(nn.Module):
    def __init__(self, latent_dim=512, n_out=2048):
        super().__init__()
        self.n_out = n_out

        self.n_seed = 256
        self.seed_points = nn.Parameter(torch.randn(1, 3, self.n_seed))
        self.seed_features = nn.Parameter(torch.randn(1, 64, self.n_seed))

        self.layer1_conv = nn.Conv1d(64, 128, kernel_size=1)
        self.layer1_gn = AdaGNBlock(128, latent_dim)

        self.expand_ratio = n_out // self.n_seed
        self.upsample_head = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, self.expand_ratio * 6, kernel_size=1),  # 6 = XYZ + Normais
        )

    def forward(self, z):
        B = z.shape[0]

        feat = self.seed_features.expand(B, -1, -1)
        pos = self.seed_points.expand(B, -1, -1)

        # Processamento com injeção do Latente Z (Estilo LION/AdaGN)
        feat = torch.gelu(self.layer1_conv(feat))
        feat = self.layer1_gn(feat, z)

        # Adensamento geométrico (Projeta para os 2048 pontos finais)
        out = self.upsample_head(feat)
        out = out.view(B, 6, self.n_out).transpose(1, 2)

        xyz = out[..., :3]
        raw_normals = torch.tanh(out[..., 3:])
        normals = raw_normals / (raw_normals.norm(dim=-1, keepdim=True) + 1e-8)

        return {"recon": torch.cat([xyz, normals], dim=-1)}
