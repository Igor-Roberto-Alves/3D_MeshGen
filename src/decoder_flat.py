import math

import torch
import torch.nn as nn

from src.Decoder import PVConvBlockDecoder, SharedMLP
from src.utils import channel_schedule


def _make_seeds(ratio: int) -> torch.Tensor:
    """
    Build a fixed (ratio, 2) 2-D seed grid covering [-1, 1]^2.
    Uses ceil(sqrt(ratio)) x ceil(sqrt(ratio)) uniform grid, cropped to ratio.
    """
    side = int(math.ceil(math.sqrt(ratio)))
    t    = torch.linspace(-1.0, 1.0, side)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    grid   = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (side^2, 2)
    return grid[:ratio]                                          # (ratio, 2)


class FlatDecoder(nn.Module):
    """
    Decoder StyleGAN-like para um latente UNICO e flat (sem split
    global/local, sem "pontos latentes").

    Ideia
    -----
    Nao existe mais um z_l por ancora. Em vez disso ha um "canvas" de
    n_latent ancoras — um nn.Parameter constante, identico para qualquer
    amostra do batch (equivalente ao "constant input" do StyleGAN). A
    UNICA fonte de variacao entre amostras e o vetor latente `z` (flat),
    injetado como estilo via AdaGN em todo estagio do decoder.

    Capacidade e 100% controlada por parametros do construtor (sem editar
    codigo): `hidden_dim` (largura inicial), `n_stages` (profundidade),
    `fold_hidden`/`fold_dim` (largura do MLP de folding) e `resolution`
    (grade de voxelizacao de cada PVConv — custo cresce ~quadratico nos
    canais mas o Conv3d por si e o item mais caro do modelo).

    Pipeline
    --------
    canvas constante (n_latent, seed_dim)
      -> posicao inicial + features (iguais p/ todo o batch)
      -> N estagios PVConv com AdaGN(z) + refino de posicao a cada estagio
         (exceto o ultimo, cuja saida alimenta o fold_mlp)
      -> fold: cada ancora x seeds 2-D -> MLP -> offset 3-D
      -> posicoes finas (n_latent * ratio, 3) = (N, 3)
    """

    def __init__(
        self,
        latent_dim:  int = 512,
        n_latent:    int = 512,
        n_points:    int = 2048,
        seed_dim:    int = 128,
        hidden_dim:  int = 384,
        n_stages:    int = 4,
        fold_dim:    int = 64,
        fold_hidden: int = 256,
        resolution:  int = 16,
    ):
        assert n_points % n_latent == 0, (
            f"n_points ({n_points}) must be divisible by n_latent ({n_latent})"
        )
        assert n_stages >= 1, "n_stages must be >= 1"
        super().__init__()
        self.n_latent = n_latent
        self.ratio    = n_points // n_latent
        self.seed_dim = seed_dim

        # Canais reais (arredondados p/ multiplo de 16, ver src.utils.channel_schedule) —
        # usados em vez dos argumentos brutos em qualquer lugar cuja shape
        # precise bater com a saida de um estagio (feat_proj e fold_mlp).
        channels = channel_schedule(hidden_dim, fold_dim, n_stages)
        hidden_dim_actual = channels[0]
        fold_dim_actual   = channels[-1]

        # "Constant input" (StyleGAN-style): aprendido, IGUAL para
        # qualquer amostra. Toda variacao entra via AdaGN(z) abaixo.
        self.anchor_embed = nn.Parameter(torch.randn(n_latent, seed_dim) * 0.02)

        self.pos_head  = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1), nn.GELU(), nn.Conv1d(64, 3, 1)
        )
        self.feat_proj = SharedMLP([seed_dim, hidden_dim_actual, hidden_dim_actual])
        self.stages = nn.ModuleList([
            PVConvBlockDecoder(channels[i], channels[i + 1], latent_dim, resolution=resolution)
            for i in range(n_stages)
        ])
        # Sem refine depois do ULTIMO estagio: a saida dele alimenta o fold_mlp direto.
        self.refines = nn.ModuleList([
            nn.Conv1d(channels[i + 1], 3, 1) for i in range(n_stages - 1)
        ])

        fold_in = fold_dim_actual + 2
        self.fold_mlp = nn.Sequential(
            nn.Linear(fold_in, fold_hidden),              nn.GELU(),
            nn.Linear(fold_hidden, fold_hidden // 2),      nn.GELU(),
            nn.Linear(fold_hidden // 2, fold_dim_actual),  nn.GELU(),
            nn.Linear(fold_dim_actual, 3),
        )

        seeds = _make_seeds(self.ratio)          # (ratio, 2) — fixo, nao-treinavel
        self.register_buffer("seeds", seeds)

    def forward(self, z: torch.Tensor, return_coarse: bool = False):
        """
        z: (B, latent_dim) — vetor latente unico (sem z_l/z_g).
        Returns: (B, n_latent * ratio, 3)
                 ou ((B, n_latent * ratio, 3), (B, n_latent, 3)) se return_coarse=True
        """
        B = z.shape[0]

        anchor   = self.anchor_embed.unsqueeze(0).expand(B, -1, -1)   # (B, n_latent, seed_dim)
        anchor_c = anchor.permute(0, 2, 1)                             # (B, seed_dim, n_latent)

        # --- Canvas inicial: igual para todo o batch (canvas constante) ---
        xyz_cur = torch.tanh(self.pos_head(anchor_c)).permute(0, 2, 1)  # (B, n_latent, 3)
        feat    = self.feat_proj(anchor_c).permute(0, 2, 1)             # (B, n_latent, hidden_dim)

        # --- A partir daqui, z (estilo) modula cada estagio via AdaGN ---
        for i, stage in enumerate(self.stages):
            feat = stage(xyz_cur, feat, z)
            if i < len(self.refines):
                xyz_cur = torch.tanh(
                    xyz_cur + self.refines[i](feat.permute(0, 2, 1)).permute(0, 2, 1)
                )

        # --- Folding: expande cada ancora em `ratio` pontos finos ---
        feat_rep  = feat.unsqueeze(2).expand(-1, -1, self.ratio, -1)
        seeds_rep = self.seeds.unsqueeze(0).unsqueeze(0).expand(B, self.n_latent, -1, -1)

        fold_in  = torch.cat([feat_rep, seeds_rep], dim=-1)
        delta    = self.fold_mlp(fold_in)

        xyz_rep  = xyz_cur.unsqueeze(2).expand(-1, -1, self.ratio, -1)
        xyz_fine = torch.tanh(xyz_rep + delta)

        xyz_fine = xyz_fine.reshape(B, self.n_latent * self.ratio, 3)
        if return_coarse:
            return xyz_fine, xyz_cur
        return xyz_fine
