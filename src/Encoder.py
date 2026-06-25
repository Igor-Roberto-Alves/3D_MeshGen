import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *




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



# Voxel branch 
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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



# Point branch
# @@@@@@@@@@@@@@@@@@@@@@@@

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
    This is the Up-side of main picture of LION model. A PVCNN to Latent Vectors
    """

    def __init__(self, in_channels: int = 6, style_dim: int = 512):
        super().__init__()

        # PVCNN
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64,          128, resolution=16)
        self.stage2 = PVConvBlock(128,         256, resolution=8)
        
        # MLP to return correct dims
        self.style_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, style_dim),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, style_dim),
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, style_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.stage0(x)               # (B, N, 64)
        x = self.stage1(x)               # (B, N, 128)
        x = self.stage2(x)               # (B, N, 256)
        

        g = x.max(dim=1).values          # Shape resultante: (B, 256)
        
        # 2. Extração dos parâmetros estatísticos da distribuição global
        mu = self.fc_mu(g)               # Shape: (B, style_dim)
        logvar = self.fc_logvar(g)       # Shape: (B, style_dim)
        
        return mu, logvar
    
    def z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
class Encoder(nn.Module):

    # This is the down-side

    def __init__(self, latent_dim: int = 512, in_channels: int = 6, style_dim: int = 512, num_latent_points: int = 512):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latent_points = num_latent_points
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64, 128, resolution=16)
        self.stage2 = PVConvBlock(128, 256, resolution=8)
        self.adagn = AdaGN(num_channels=256, style_dim=style_dim, num_groups=32)

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


        self.fc_mu     = nn.Conv1d(512, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(512, latent_dim, kernel_size=1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # x ~ (B, 2048, 6) x,y,z,n_1,n_2,n_3
        xyz_original = x[:, :, :3].clone() # WAITING WAITING
        x = self.stage0(x)               # (B, 2048, 64)
        x = self.stage1(x)               # (B, 2048, 128)
        x = self.stage2(x)               # (B, 2048, 256)

        
        # Permutamos para (B, Canais, Pontos) pois o GroupNorm interno do AdaGN exige este formato
        x = x.permute(0, 2, 1)           # (B, 256, 2048)

        

        x = self.adagn(x, style)         # (B, 256, N_original)
        
        x = x.permute(0, 2, 1)           # (B, 2048, 256)

        # fps downsampling

        fps_indices = farthest_point_sample(xyz_original, self.num_latent_points)
        x_latent = index_points(x, fps_indices)
        latent_xyz = index_points(xyz_original, fps_indices)


        x_latent = self.local_mlp(x_latent.permute(0, 2, 1)).permute(0, 2, 1)  # (B, num_latent_points, 512)

        # Permutamos para rodar as convoluções de predição Conv1d (B, Canais, Pontos)
        x_features = x_latent.permute(0, 2, 1) # (B, 512, num_latent_points)
        
        
        mu_feat     = self.fc_mu(x_features).permute(0, 2, 1)     
        logvar_feat = self.fc_logvar(x_features).permute(0, 2, 1)  

      
        mu = torch.cat([latent_xyz, mu_feat], dim=-1) # Shape final: (B, num_latent_points, 3 + latent_dim)
        
        zeros_padding = torch.full_like(latent_xyz, -4)
        logvar = torch.cat([zeros_padding, logvar_feat], dim=-1) # Shape final: (B, num_latent_points, 3 + latent_dim)

        return mu, logvar
    

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 1, in_channels: int = 6, style_dim: int = 512, num_latent_points = 2048):
        super().__init__()
        # 1. Mudança LION: latent_dim do ponto passa a ser 1 em vez de 512
        self.latent_dim = latent_dim 
        
        self.stage0 = PVConvBlock(in_channels, 64,  resolution=32)
        self.stage1 = PVConvBlock(64, 128, resolution=16)
        self.stage2 = PVConvBlock(128, 256, resolution=8)
        
        # Camada AdaGN para injetar o estilo global nas features locais
        self.adagn = AdaGN(num_channels=256, style_dim=style_dim, num_groups=32)

        # Processamento local pós-AdaGN
        self.local_mlp = SharedMLP([256, 512])

        # Cabeças de predição estatística (LION opera por ponto via Conv1d)
        # fc_mu agora prevê 3 canais para o Delta XYZ + 1 canal para a Feature Latente = 4
        self.fc_mu     = nn.Conv1d(512, 3 + self.latent_dim, kernel_size=1)
        # fc_logvar prevê a variância para as coordenadas e para a feature = 4
        self.fc_logvar = nn.Conv1d(512, 3 + self.latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, 2048, 6) -> [xyz | normals]
        xyz_original = x[:, :, :3].clone().contiguous() 
        
        # Passagem pelo esqueleto PVCNN
        x = self.stage0(x)               # (B, 2048, 64)
        x = self.stage1(x)               # (B, 2048, 128)
        x = self.stage2(x)               # (B, 2048, 256)

        # Condicionamento AdaGN com o Estilo Global
        x = x.permute(0, 2, 1)           # (B, 256, 2048)
        x = self.adagn(x, style)         # (B, 256, 2048)
        
        # Expansão de canais locais
        x_latent = self.local_mlp(x)     # (B, 512, 2048)

        # Predição dos parâmetros da distribuição Gaussiana
        mu_raw = self.fc_mu(x_latent).permute(0, 2, 1)         # (B, 2048, 3 + latent_dim)
        logvar_raw = self.fc_logvar(x_latent).permute(0, 2, 1) # (B, 2048, 3 + latent_dim)

        # -------------------------------------------------------------------
        # A MÁGICA DO LION: CONEXÕES RESIDUAIS PONDERADAS (Weighted Skips)
        # -------------------------------------------------------------------
        # Separamos o delta previsto para o XYZ e a média da feature
        mu_xyz_delta = mu_raw[:, :, :3]
        mu_feat = mu_raw[:, :, 3:]

        # A média das coordenadas latentes NÃO é do zero: é a original + delta amortecido por 0.01
        mu_xyz = xyz_original + (0.01 * mu_xyz_delta)
        
        # Juntamos de volta para formar o MU final do VAE
        mu = torch.cat([mu_xyz, mu_feat], dim=-1) # (B, 2048, 4)

        # -------------------------------------------------------------------
        # VARIANCE SCALING: LOG_SIGMA_OFFSET = 6.0
        # -------------------------------------------------------------------
        # Subtraímos um offset de 6.0 (ou outro valor alto) para forçar a variância 
        # inicial a ser extremamente próxima de zero, agindo como uma identidade.
        logvar = logvar_raw - 6.0 # Aplica-se tanto a XYZ como às features!

        return mu, logvar