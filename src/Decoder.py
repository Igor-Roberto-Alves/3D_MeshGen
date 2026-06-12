import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Encoder import *

# ---------------------------------------------------------------------------
# Módulo de Modulação de Estilo (AdaGN / MLP Modulation)
# ---------------------------------------------------------------------------
class StyleModulation(nn.Module):
    def __init__(self, style_dim: int, feat_channels: int):
        super().__init__()
        self.to_scale = nn.Linear(style_dim, feat_channels)
        self.to_shift = nn.Linear(style_dim, feat_channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale = torch.tanh(self.to_scale(style)).unsqueeze(1)
        shift = 0.1 * torch.tanh(self.to_shift(style)).unsqueeze(1)
        return x * (1 + scale) + shift



class PointUpsampleBlock(nn.Module):
    """
    Upsampling learned: N → kN
    cada ponto gera k offsets
    """
    def __init__(self, in_channels: int, k: int = 4):
        super().__init__()
        self.k = k

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.GELU(),
            nn.Conv1d(in_channels, 3 * k, 1)  # offsets XYZ
        )

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """
        xyz:  (B, N, 3)
        feat: (B, N, C)
        return (B, N*k, 3)
        """

        B, N, _ = xyz.shape

        x = feat.permute(0, 2, 1)          # (B, C, N)
        offsets = self.mlp(x)              # (B, 3k, N)

        offsets = offsets.view(B, self.k, 3, N)
        offsets = offsets.permute(0, 3, 1, 2)   # (B, N, k, 3)

        xyz = xyz.unsqueeze(2)             # (B, N, 1, 3)

        new_xyz = xyz + offsets            # (B, N, k, 3)

        return new_xyz.reshape(B, N * self.k, 3)
    
# ---------------------------------------------------------------------------
# LION Decoder Core - CORRIGIDO SEGUNDO O PADRÃO NVIDIA LION
# ---------------------------------------------------------------------------
class LIONDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,   # Dimensão das características abstratas (mochila)
        style_dim: int = 512,    # Dimensão do vetor de estilo global (z0)
        input_dim: int = 3,       # Coordenadas físicas da entrada (XYZ = 3 ou XYZ+Normais = 6)
        up_factor: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim

        # MUDANÇA 1: A entrada real que vem do espaço latente unificado da NVIDIA
        # contém as coordenadas espaciais + as feições grudadas. Total = 3 + 256 = 259 canais.
        total_in_channels = latent_dim + input_dim
        
        # Projetamos essa mistura de canais para o tamanho inicial do bloco PVCNN (256)
        self.input_projection = SharedMLP([total_in_channels, 256])

        # Módulos de Modulação Adaptativa de Estilo
        self.mod0 = StyleModulation(style_dim, 256)
        self.mod1 = StyleModulation(style_dim, 128)

        # Blocos PVCNN Reversos
        self.stage0 = PVConvBlock(256, 128, resolution=8)
        self.stage1 = PVConvBlock(128, 64,  resolution=16)
        self.stage2 = PVConvBlock(64,  32,  resolution=32)
        self.upsample = PointUpsampleBlock(32, k=up_factor)
        # Cabeçalho de saída ponto a ponto
        # Fornecerá os deltas de deslocamento e as novas normais
        self.output_head = nn.Sequential(
            nn.Conv1d(32, 16, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 6, 1) # Cospe 6 canais: 3 para delta XYZ + 3 para Normais
        )

    def forward(self, latent_points: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        latent_points : (B, N, latent_dim + input_dim) -> Ex: (B, 128, 259)
        style         : (B, style_dim)
        """
        # --- BASE FÍSICA ---
        base_coords = latent_points[:, :, :3]  # (B, N, 3)

        # --- PROJEÇÃO INICIAL ---
        x = latent_points.permute(0, 2, 1)
        x = self.input_projection(x).permute(0, 2, 1)      # (B, N, 256)

        # --- MODULAÇÃO + PVCNN ---
        x = self.mod0(x, style)
        x = self.stage0(x)

        x = self.mod1(x, style)
        x = self.stage1(x)

        x = self.stage2(x)  # (B, N, 32)

        # --- UPSAMPLING 128 -> 2048 ---
        coords_up = self.upsample(base_coords, x)  # (B, N*4, 3)
        
        # Expand features para cada ponto novo
        B, N, C = x.shape
        x_up = x.unsqueeze(2).repeat(1, 1, 4, 1)      # (B, N, 4, 32)
        x_up = x_up.reshape(B, N*4, C)                # (B, N*4, 32)

        # --- HEAD FINAL ---
        feat_out = self.output_head(x_up.permute(0, 2, 1))  # (B, 6, N*4)
        feat_out = feat_out.permute(0, 2, 1)                # (B, N*4, 6)

        # --- DELTA + NORMAL ---
        delta_coords = 0.5 * torch.tanh(feat_out[:, :, :3])
        coords = coords_up + delta_coords
        coords = torch.tanh(coords)

        normals = F.normalize(feat_out[:, :, 3:], p=2, dim=-1)
        coords = torch.clamp(coords, -1, 1)
        return coords, normals