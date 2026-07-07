import torch


def round_channels(x: float, base: int = 16, min_val: int = 16) -> int:
    return max(min_val, int(round(x / base)) * base)


def channel_schedule(start_dim: int, end_dim: int, n_steps: int) -> list[int]:
    """
    Interpolacao geometrica de `start_dim` ate `end_dim` em `n_steps`
    passos, com cada valor arredondado pra um multiplo de 16 (ver
    `round_channels`). Retorna uma lista de `n_steps + 1` valores;
    channels[i] -> channels[i+1] e o i-esimo estagio.
    """
    channels = []
    for i in range(n_steps + 1):
        r = i / n_steps
        c = start_dim * ((end_dim / start_dim) ** r)
        channels.append(round_channels(c))
    channels[0]  = round_channels(start_dim)
    channels[-1] = round_channels(end_dim)
    return channels

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    device = xyz.device

    # 🔒 proteção contra NaN/INF
    xyz = torch.nan_to_num(xyz, nan=0.0, posinf=1e3, neginf=-1e3)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    # Início DETERMINÍSTICO: ponto mais distante do centróide.
    # Antes era torch.randint (aleatório), o que fazia o subconjunto de âncoras
    # — e portanto o latente z_l — variar a cada chamada para a mesma nuvem.
    # Na difusão isso injetava ruído no alvo (z_l extraído a cada step muda por
    # época). Com um seed determinístico e canônico (independe da ordem em que
    # os pontos estão salvos), o mesmo shape gera sempre o mesmo z_l.
    centroid = xyz.mean(dim=1, keepdim=True)             # (B, 1, 3)
    farthest = ((xyz - centroid) ** 2).sum(dim=-1).argmax(dim=1)  # (B,)

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
        
        # MLP que transforma o estilo global nos parâmetros lineares (gamma e beta).
        # Peso com init PEQUENO mas nao-zero: com peso exatamente zero,
        # d(saida)/d(style) == 0 no passo 0, entao NENHUM gradiente de
        # reconstrucao chega ao encoder enquanto o decoder ja treina — num
        # decoder de canvas constante (FlatDecoder) o AdaGN e o UNICO caminho
        # de z ate a saida, e esse caminho morto no init fazia o decoder
        # convergir pra "forma media" incondicional antes do encoder acordar
        # (posterior collapse ja na fase beta=0). std=0.01 mantem gamma/beta
        # ~N(0, 0.23^2) p/ style_dim=512: perto da identidade, mas vivo.
        self.fc = nn.Linear(style_dim, num_channels * 2)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (B, num_channels, N) - Formato padrão de Conv1d / GroupNorm
        # style: (B, style_dim)
        
        # 1. Aplica a normalização por grupo padrão
        x_norm = self.gn(x)
        
        # 2. Gera os coeficientes a partir do estilo
        style_effects = self.fc(style).unsqueeze(-1) # (B, num_channels * 2, 1)
        gamma, beta = torch.chunk(style_effects, 2, dim=1) # Divide ao meio
        
        # Como inicializamos os pesos em zero, somamos 1 ao gamma 
        # para que o comportamento inicial seja multiplicar por 1 (identidade)
        return x_norm * (1 + gamma) + beta