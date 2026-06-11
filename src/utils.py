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