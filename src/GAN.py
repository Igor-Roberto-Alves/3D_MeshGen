import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetDiscriminator(nn.Module):
    """
    Discriminador estilo PointNet para nuvens de pontos (B, N, 6).

    Processa cada ponto independentemente via shared MLP, agrega com
    max-pooling global e classifica real/fake com um MLP final.

    Args:
        in_ch       : canais de entrada por ponto (6 = XYZ + normais)
        base_ch     : largura base dos canais internos
        use_spectral: aplica spectral norm em todas as camadas lineares
                      (estabiliza treino GAN sem precisar de batch norm)
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 64, use_spectral: bool = True):
        super().__init__()

        def linear(a, b):
            l = nn.Linear(a, b)
            return nn.utils.spectral_norm(l) if use_spectral else l

        # Shared MLP ponto-a-ponto: (B, N, in_ch) → (B, N, base_ch*8)
        self.mlp = nn.Sequential(
            linear(in_ch, base_ch),  # 6   → 64
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch, base_ch * 2),  # 64  → 128
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 2, base_ch * 4),  # 128 → 256
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 4, base_ch * 8),  # 256 → 512
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classificador global: (B, base_ch*8) → (B, 1)
        self.head = nn.Sequential(
            linear(base_ch * 8, base_ch * 4),  # 512 → 256
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 4, base_ch),  # 256 → 64
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch, 1),  # 64  → 1 (logit)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts : (B, N, 6)
        retorna logits (B, 1) — sem sigmoid, compatível com BCEWithLogitsLoss
        """
        feat = self.mlp(pts)  # (B, N, base_ch*8)
        feat = feat.max(dim=1).values  # (B, base_ch*8) — global max pool
        return self.head(feat)  # (B, 1)
