import torch

import torch

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    device = xyz.device

    # 🔒 proteção contra NaN/INF
    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=1e3, neginf=-1e3)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    # inicialização mais estável
    farthest = torch.randint(0, N, (B,), device=device)

    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest].unsqueeze(1)

        diff = xyz - centroid
        dist = torch.sum(diff * diff, dim=-1)

        # 🔒 proteção hard
        dist = torch.nan_to_num(dist, nan=1e10, posinf=1e10)

        # update estável (evita boolean indexing)
        distance = torch.minimum(distance, dist)

        farthest = torch.argmax(distance, dim=-1)

    return centroids

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Filtra a nuvem de pontos original usando os índices gerados pelo FPS."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

from torch import nn
class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.
    Normaliza os recursos dos pontos e aplica escala (gamma) e translação (beta)
    gerados dinamicamente a partir do vetor de estilo global.
    """
    def __init__(self, num_channels: int, style_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_channels = num_channels
        # GroupNorm padrão (filtros, canais, eps)
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=1e-5)
        
        # MLP que transforma o estilo global nos parâmetros lineares (gamma e beta)
        # Inicializamos o peso da escala em zero para começar como uma identidade
        self.fc = nn.Linear(style_dim, num_channels * 2)
        # Paper D.1: weight scale 0.1, output factor bias=1.0, output shift bias=0.0
        nn.init.normal_(self.fc.weight, std=0.1)
        nn.init.ones_(self.fc.bias[:num_channels])   # factor starts at 1 (identity scale)
        nn.init.zeros_(self.fc.bias[num_channels:])  # shift starts at 0

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (B, num_channels, N) - Formato padrão de Conv1d / GroupNorm
        # style: (B, style_dim)
        
        # 1. Aplica a normalização por grupo padrão
        x_norm = self.gn(x)
        
        # 2. Gera os coeficientes a partir do estilo
        style_effects = self.fc(style).unsqueeze(-1)          # (B, num_channels*2, 1)
        gamma, beta   = torch.chunk(style_effects, 2, dim=1)  # factor, shift
        # gamma bias initialised to 1 → identity scale at init; beta bias 0 → no shift
        return x_norm * gamma + beta


def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, K: int) -> torch.Tensor:
    """
    xyz:     (B, N, 3) — all points
    new_xyz: (B, M, 3) — query centers (FPS output)
    Returns: (B, M, K) indices into xyz for the K nearest neighbors within radius.
    If fewer than K neighbors exist within radius, repeats the nearest one.
    """
    dist = torch.cdist(new_xyz.float(), xyz.float())   # (B, M, N)
    dist_masked = dist.clone()
    dist_masked[dist > radius] = 1e10
    _, idx = dist_masked.topk(K, dim=-1, largest=False)  # (B, M, K)
    nearest = idx[..., :1].expand_as(idx)
    mask = (dist_masked.gather(2, idx) >= 1e10)
    idx = torch.where(mask, nearest, idx)
    return idx


class SqueezeExcitation(nn.Module):
    """Channel-wise attention (Hu et al. 2018). Applied to (B, C, N) tensors."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)              # (B, C) global average
        w = self.net(s).unsqueeze(-1)   # (B, C, 1)
        return x * w


class PointSelfAttention(nn.Module):
    """
    Lightweight self-attention over M points.
    Input/output: (B, M, C)
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)