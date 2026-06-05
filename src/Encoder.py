import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

class SharedMLP(nn.Sequential):
    """1-D shared MLP applied point-wise (B, C, N) → (B, C', N)."""
    def __init__(self, channels: list[int], bn: bool = True, act: bool = True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            if act:
                layers.append(nn.GELU())
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# Voxel branch (PVCNN-style)
# ---------------------------------------------------------------------------

class VoxelBranch(nn.Module):

    """
    Voxelises a point cloud, applies 3-D convolutions, then trilinearly
    samples the voxel features back onto the original point positions.

    Args:
        in_channels  : number of input point features (e.g. 6 for xyz+normals)
        out_channels : number of output voxel features per point
        resolution   : voxel grid resolution (cubic)
    """

    def __init__(self, in_channels: int, out_channels: int, resolution: int = 16):
        super().__init__()
        self.resolution = resolution
        mid = out_channels * 2

        self.voxel_net = nn.Sequential(
            nn.Conv3d(in_channels, mid, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid),
            nn.GELU(),
            nn.Conv3d(mid, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    def _voxelise(self, coords: torch.Tensor, features: torch.Tensor,
                  R: int) -> torch.Tensor:
        
        """
        Scatter-average features into a (B, C, R, R, R) voxel grid.

        coords   : (B, 3, N)  – already normalised to [0, R-1]
        features : (B, C, N)
        """

        B, C, N = features.shape
        device = features.device

        idx = coords.long().clamp(0, R - 1)          # (B, 3, N)
        flat_idx = idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2]  # (B, N)

        # accumulate sum + count
        voxels = torch.zeros(B, C, R * R * R, device=device, dtype = features.dtype)
        count  = torch.zeros(B, 1, R * R * R, device=device, dtype = features.dtype)

        flat_idx_exp = flat_idx.unsqueeze(1).expand_as(features)  # (B, C, N)
        voxels.scatter_add_(2, flat_idx_exp, features)

        count_src = torch.ones(B, 1, N, device=device, dtype = features.dtype)
        count.scatter_add_(2, flat_idx.unsqueeze(1), count_src)

        count = count.clamp(min=1.0)
        voxels = voxels / count                                    # average

        return voxels.view(B, C, R, R, R)

    # ------------------------------------------------------------------
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points : (B, N, 6)  – [xyz | normals]
        returns: (B, N, out_channels)
        """
        B, N, _ = points.shape
        R = self.resolution

        coords   = points[:, :, :3].permute(0, 2, 1)   # (B, 3, N)
        features = points.permute(0, 2, 1)              # (B, 6, N)

        # normalise xyz to voxel grid indices
        xyz_min = coords.min(dim=2, keepdim=True).values
        xyz_max = coords.max(dim=2, keepdim=True).values
        scale   = (xyz_max - xyz_min).clamp(min=1e-4)
        norm_coords = (coords - xyz_min) / scale * (R - 1)  # [0, R-1]

        # voxelise → 3-D conv → sample back
        voxel_grid   = self._voxelise(norm_coords, features, R)   # (B, C, R, R, R)
        voxel_feats  = self.voxel_net(voxel_grid)                 # (B, out_C, R, R, R)

        # trilinear sampling
        sample_grid  = norm_coords / (R - 1) * 2 - 1             # [-1, 1]
        sample_grid  = torch.clamp(sample_grid, min=-1.0, max=1.0)
        sample_grid  = sample_grid.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
        # grid_sample expects (B, D_out, H_out, W_out, 3)
        sample_grid  = sample_grid.expand(-1, 1, 1, N, 3)        # (B, 1, 1, N, 3)
        

        sampled = F.grid_sample(
            voxel_feats, sample_grid,
            mode="bilinear", align_corners=True, padding_mode="border"
        )                                                          # (B, out_C, 1, 1, N)
        sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1) # (B, N, out_C)
        return sampled


# ---------------------------------------------------------------------------
# Point branch (PointNet-style)
# ---------------------------------------------------------------------------

class PointBranch(nn.Module):
    """Lightweight point-wise MLP branch."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = SharedMLP([in_channels, out_channels * 2, out_channels])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, C) → (B, N, out_channels)"""
        x = points.permute(0, 2, 1)       # (B, C, N)
        x = self.mlp(x)
        return x.permute(0, 2, 1)         # (B, N, out_channels)


# ---------------------------------------------------------------------------
# PVCNN Block  (voxel + point fusion)
# ---------------------------------------------------------------------------

class PVConvBlock(nn.Module):
    """
    One PVCNN-style block: fuses a voxel branch with a point branch.

    in_channels  : input feature dim per point
    out_channels : output feature dim per point
    resolution   : voxel resolution for this stage
    """

    def __init__(self, in_channels: int, out_channels: int, resolution: int = 16):
        super().__init__()
        self.voxel = VoxelBranch(in_channels, out_channels // 2, resolution)
        self.point = PointBranch(in_channels, out_channels // 2)

        # fusion + projection
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, in_channels) → (B, N, out_channels)"""
        v = self.voxel(points)             # (B, N, out//2)
        p = self.point(points)             # (B, N, out//2)
        x = torch.cat([v, p], dim=2)      # (B, N, out)
        x = self.fuse(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderStyle(nn.Module):
    """
    Este bloco recebe a nuvem bruta (B, N, 6), passa pelas convoluções
    e colapsa os pontos para gerar o vetor de estilo global.
    """
    def __init__(self, in_channels: int = 6, style_dim: int = 512):
        super().__init__()
        # Usamos os mesmos blocos PVCNN para entender a geometria
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64,          128, resolution=16)
        self.stage2 = PVConvBlock(128,         256, resolution=8)
        
        # Uma MLP que vai moldar o vetor após o Max-Pooling
        self.style_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 6)
        x = self.stage0(x)               # (B, N, 64)
        x = self.stage1(x)               # (B, N, 128)
        x = self.stage2(x)               # (B, N, 256)
        
        # Max-pool sobre a dimensão dos pontos para torná-lo GLOBAL
        g = x.max(dim=1).values          # (B, 256)
        
        # Gera o vetor de estilo (B, style_dim)
        return self.style_mlp(g)
    
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 256, in_channels: int = 6, style_dim: int = 512, num_latent_points: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latent_points = num_latent_points
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64,          128, resolution=16)
        self.stage2 = PVConvBlock(128,         256, resolution=8)

        # Projeta o estilo global de volta para os 256 canais que saem do stage2
        self.style_projection = nn.Linear(style_dim, 256)

        # Recebe os 256 canais (já misturados com o estilo) e expande para 512
        self.local_mlp = SharedMLP([256, 512])

        # global aggregation head
        self.global_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )

        #self.fc_mu     = nn.Linear(512, latent_dim)
        #self.fc_logvar = nn.Linear(512, latent_dim)
        self.fc_mu     = nn.Conv1d(512, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(512, latent_dim, kernel_size=1)

    # ------------------------------------------------------------------
    # CORREÇÃO 1: Adicionado o parâmetro 'style' na assinatura
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        xyz = x[:, :, :3] # Shape: (B, N_original, 3)
        fps_indices = farthest_point_sample(xyz, self.num_latent_points) # (B, num_latent_points)
        
        # Filtra a nuvem original usando os índices, mantendo todos os 6 canais (XYZ + Normais)
        x = index_points(x, fps_indices) # Agora x passa a ter formato: (B, num_latent_points, 6)

        # --- NOVA LINHA MANDATÓRIA: Guarda o XYZ filtrado pelo FPS para colar no final ---
        latent_xyz = x[:, :, :3] # Shape: (B, num_latent_points, 3)

        x = self.stage0(x)               # (B, N, 64)
        x = self.stage1(x)               # (B, N, 128)
        x = self.stage2(x)               # (B, N, 256)

        s = self.style_projection(style) # (B, 256)
        x = x + s.unsqueeze(1)           # (B, N, 256)

        x = self.local_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 512)

        # Permutamos para (B, Canais, Pontos) pois nn.Conv1d opera no eixo 1
        x_features = x.permute(0, 2, 1) # (B, 512, N)
        
        # Mapeia os 512 canais de cada ponto para a dimensão latente desejada (256 canais)
        mu_feat     = self.fc_mu(x_features).permute(0, 2, 1)     # (B, N, latent_dim)
        logvar_feat = self.fc_logvar(x_features).permute(0, 2, 1)  # (B, N, latent_dim)

        # --- AJUSTE DE ARQUITETURA LION (NVIDIA) ---
        # Colamos o latent_xyz físico na frente do mu (3 + 256 = 259 canais)
        mu = torch.cat([latent_xyz, mu_feat], dim=-1) # Shape: (B, N, 259)
        
        # Para o logvar, colocamos variância zero (zeros) na parte das coordenadas fixas
        zeros_padding = torch.zeros_like(latent_xyz)
        logvar = torch.cat([zeros_padding, logvar_feat], dim=-1) # Shape: (B, N, 259)

        return mu, logvar