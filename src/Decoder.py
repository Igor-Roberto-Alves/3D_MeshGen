import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Encoder import *
# ---------------------------------------------------------------------------
# Módulo de Modulação de Estilo (AdaGN / MLP Modulation)
# ---------------------------------------------------------------------------
class StyleModulation(nn.Module):
    """
    Modula as feições dos pontos locais usando o vetor de estilo global.
    Aplica: out = x * scale(style) + shift(style)
    """
    def __init__(self, style_dim: int, feat_channels: int):
        super().__init__()
        self.to_scale = nn.Linear(style_dim, feat_channels)
        self.to_shift = nn.Linear(style_dim, feat_channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (B, N, feat_channels) | style: (B, style_dim)
        scale = torch.tanh(self.to_scale(style)).unsqueeze(1)
        shift = 0.1 * torch.tanh(self.to_shift(style)).unsqueeze(1)

        return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# LION Decoder Core
# ---------------------------------------------------------------------------
class LIONDecoder(nn.Module):
    """
    LION Decoder oficial baseado em pontos condicionais.
    Transforma pontos latentes h0 (B, N, latent_dim) guiados por um 
    vetor de estilo global z0 (B, style_dim) na nuvem 3D final com normais.

    Arquitetura Reversa
    -------------------
    Stage 0 : Modulação inicial + PVConvBlock 256 → 128 (res=8)
    Stage 1 : Modulação intermediária + PVConvBlock 128 → 64  (res=16)
    Stage 2 : PVConvBlock 64 → 32  (res=32)
    Head    : SharedMLP 32 → 6 (XYZ + Normais)
    """
    def __init__(
        self,
        latent_dim: int = 256,   # dimensão dos pontos latentes locais (h0)
        style_dim: int = 512,    # dimensão do vetor de estilo global (z0)
        out_channels: int = 6     # XYZ (3) + Normais (3)
    ):
        super().__init__()

        # Camada de projeção inicial para alinhar os canais latentes locais
        self.input_projection = SharedMLP([latent_dim, 256])

        # Módulos de Modulação Adaptativa de Estilo para cada estágio do Decoder
        self.mod0 = StyleModulation(style_dim, 256)
        self.mod1 = StyleModulation(style_dim, 128)

        # Blocos PVCNN Reversos (Up-sampling / Refinamento hierárquico)
        # Nota: No decoder, as resoluções dos voxels aumentam para suavizar a topologia
        self.stage0 = PVConvBlock(256, 128, resolution=8)
        self.stage1 = PVConvBlock(128, 64,  resolution=16)
        self.stage2 = PVConvBlock(64,  32,  resolution=32)

        # Cabeçalho de saída ponto a ponto
        self.output_head = nn.Sequential(
            nn.Conv1d(32, 16, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, out_channels, 1) # Projeta para XYZ + Normais
        )

    def forward(self, latent_points: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        latent_points : (B, N, latent_dim) -> h0 gerado pelo modelo de difusão
        style         : (B, style_dim)     -> z0 gerado pelo modelo de difusão global

        Returns
        -------
        coords  : (B, N, 3) – Coordenadas XYZ finais reconstruídas
        normals : (B, N, 3) – Vetores normais da superfície calculados por ponto
        """
        # 1. Projeção Inicial dos canais latentes
        # Passa de (B, N, latent_dim) -> (B, N, 256)
        x = latent_points.permute(0, 2, 1)
        x = self.input_projection(x).permute(0, 2, 1)

        # 2. Estágio 0: Modula com o Estilo + Processamento PVCNN
        x = self.mod0(x, style)
        x = self.stage0(x)               # (B, N, 128)

        # 3. Estágio 1: Segunda rodada de Modulação + Processamento PVCNN
        x = self.mod1(x, style)
        x = self.stage1(x)               # (B, N, 64)

        # 4. Estágio 2: Refinamento Geométrico Final de Alta Resolução
        x = self.stage2(x)               # (B, N, 32)

        # 5. Projeção para o espaço 3D físico
        # Transforma os 32 canais abstratos em 6 canais reais (B, 6, N)
        feat_out = self.output_head(x.permute(0, 2, 1))
        feat_out = feat_out.permute(0, 2, 1) # Retorna para (B, N, 6)

        # Separando o output de coordenadas físicas e vetores normais
        coords  = feat_out[:, :, :3]    # Primeiras 3 colunas: X, Y, Z
        normals = feat_out[:, :, 3:]    # Últimas 3 colunas: Nx, Ny, Nz
        coords = torch.tanh(feat_out[:, :, :3])
        # Força as normais a possuírem comprimento unitário matemático (Normalização L2)
        normals = F.normalize(normals, p=2, dim=-1)

        return coords, normals