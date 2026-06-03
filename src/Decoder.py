import torch
from torch import nn


class PointDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        d_model: int = 256,  # Mantido na assinatura para não quebrar o seu Vae.py
        n_out: int = 2048,
    ):
        super().__init__()
        self.n_out = n_out

        # Um MLP robusto de alta capacidade para inflar o vetor Z direto para a nuvem
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            # Saída direta: Total de pontos (2048) * 6 canais (X, Y, Z, nx, ny, nz)
            nn.Linear(4096, n_out * 6),
            nn.Tanh(),
        )

    def forward(self, z):
        B = z.shape[0]

        # 1. Projeção direta via força bruta do MLP
        out = self.mlp(z)  # Shape: (B, n_out * 6)

        # 2. Reshape para o formato padrão da sua Loss e do Discriminator
        out = out.view(B, self.n_out, 6)  # Shape: (B, 2048, 6)

        # 3. Separar as coordenadas e as normais
        xyz = out[..., :3]
        raw_normals = out[..., 3:]

        # 4. Normalizar os vetores normais para mantê-los unitários (importante para a Loss de Normais)
        normals = raw_normals / (raw_normals.norm(dim=-1, keepdim=True) + 1e-8)

        return {"recon": torch.cat([xyz, normals], dim=-1)}
