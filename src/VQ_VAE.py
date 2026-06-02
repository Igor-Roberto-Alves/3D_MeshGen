import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.Tokenizer import (
    PatchEmbed,
    PositionalEncoding,
    farthest_point_sampling,
    knn_group,
)
from src.Encoder import (
    TransformerBlock,
    HierarchicalEncoder,
    GlobalEncoder,
    CrossBranchFusion,
)
from src.Decoder import PointDecoder
from src.metric import chamfer_distance

# ── VQ Bottleneck ─────────────────────────────────────────────────────────────


class VQBottleneck(nn.Module):
    """
    Vector Quantization bottleneck (van den Oord et al., 2017).

    Projeta o encoding contínuo para `latent_dim`, encontra o vetor mais
    próximo no codebook e devolve o quantizado com gradiente via straight-
    through estimator.

    Args:
        in_dim       : dimensão de entrada (d_model do encoder)
        latent_dim   : dimensão de cada vetor no codebook (= embedding_dim)
        n_embeddings : tamanho do codebook (número de vetores discretos)
        commitment_cost : peso β da commitment loss (padrão 0.25)
        ema_update   : usa EMA para atualizar codebook (mais estável que gradiente)
        ema_decay    : fator de decaimento para EMA (γ, padrão 0.99)
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_embeddings: int = 512,
        commitment_cost: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.commitment_cost = commitment_cost
        self.ema_update = ema_update

        # Projeção: d_model → latent_dim
        self.proj_in = nn.Linear(in_dim, latent_dim)

        # Codebook
        self.embedding = nn.Embedding(n_embeddings, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1 / n_embeddings, 1 / n_embeddings)

        # EMA (atualização estável do codebook sem gradiente direto nele)
        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
            self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
            self.ema_decay = ema_decay

    # ── quantização ───────────────────────────────────────────────────────────
    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_e : (B, latent_dim)
        Retorna (z_q, indices) onde z_q tem mesmo shape que z_e.
        """
        # Distâncias ao codebook: ||z_e - e_k||²
        # = ||z_e||² + ||e_k||² - 2 * z_e @ e_k^T
        d = (
            z_e.pow(2).sum(1, keepdim=True)  # (B, 1)
            + self.embedding.weight.pow(2).sum(1)  # (K,)
            - 2 * z_e @ self.embedding.weight.T  # (B, K)
        )  # (B, K)
        indices = d.argmin(1)  # (B,)
        z_q = self.embedding(indices)  # (B, latent_dim)
        return z_q, indices

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        """Atualiza codebook via EMA durante o training."""
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.n_embeddings).float()  # (B, K)
            cluster_size = one_hot.sum(0)  # (K,)
            embed_sum = one_hot.T @ z_e  # (K, D)

            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            self.ema_embed_sum.mul_(self.ema_decay).add_(
                embed_sum, alpha=1 - self.ema_decay
            )

            # Laplace smoothing para evitar vetores mortos
            n = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + 1e-5) / (n + self.n_embeddings * 1e-5) * n
            )
            self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h : (B, in_dim)  — saída do CrossBranchFusion
        Retorna (z_q_st, codebook_loss, indices)
          z_q_st      : quantizado com straight-through gradient  (B, latent_dim)
          codebook_loss: escalar (commitment + codebook)
          indices      : índices discretos selecionados           (B,)
        """
        z_e = self.proj_in(h)  # (B, latent_dim)
        z_q, indices = self.quantize(z_e)

        if self.training and self.ema_update:
            self._ema_update(z_e.detach(), indices)
            # Com EMA o codebook não recebe gradiente — só commitment loss
            codebook_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            # Sem EMA: gradiente flui tanto pro encoder quanto pro codebook
            codebook_loss = F.mse_loss(
                z_e.detach(), z_q
            ) + self.commitment_cost * F.mse_loss(  # codebook loss
                z_e, z_q.detach()
            )  # commitment

        # Straight-through: gradiente passa direto de z_q para z_e
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, codebook_loss, indices

    # ── utils ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Converte índices discretos → vetores do codebook."""
        return self.embedding(indices)

    @property
    def codebook(self) -> torch.Tensor:
        return self.embedding.weight


# ── Modelo Principal ──────────────────────────────────────────────────────────


class DualBranchPointVQVAE(nn.Module):
    """
    Dual-branch VQ-VAE para nuvens de pontos.

    Diferenças em relação ao VAE contínuo:
    - Sem μ/σ — o espaço latente é discreto (índices no codebook)
    - KL loss substituída por commitment loss + codebook loss
    - `sample` sorteia índices aleatórios do codebook
    - `interpolate` faz interpolação no espaço contínuo pré-quantização

    Args:
        d_model        : dimensão interna do transformer
        latent_dim     : dimensão de cada vetor do codebook
        n_embeddings   : número de entradas no codebook
        n_out          : pontos gerados na reconstrução
        enc_depth      : camadas transformer no encoder
        dec_depth      : camadas transformer no decoder
        commitment_cost: peso da commitment loss (β do VQ-VAE)
        ema_update     : True para atualizar codebook via EMA
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 512,
        n_embeddings: int = 512,
        n_out: int = 2048,
        enc_depth: int = 4,
        dec_depth: int = 4,
        n_heads: int = 6,
        commitment_cost: float = 0.25,
        ema_update: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings

        self.hier_enc = HierarchicalEncoder(
            d_model=d_model, n_heads=n_heads, depth=enc_depth
        )
        self.glob_enc = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion = CrossBranchFusion(d_model=d_model, n_heads=n_heads)

        # ← única mudança estrutural relevante em relação ao VAE
        self.bottleneck = VQBottleneck(
            in_dim=d_model,
            latent_dim=latent_dim,
            n_embeddings=n_embeddings,
            commitment_cost=commitment_cost,
            ema_update=ema_update,
        )

        self.decoder = PointDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=dec_depth,
            n_out=n_out,
        )

    # ── encode / decode ───────────────────────────────────────────────────────
    def encode(self, xyz: torch.Tensor):
        """
        Retorna (z_q_st, codebook_loss, indices).
        Durante o training use z_q_st para backprop.
        """
        h_feat, _ = self.hier_enc(xyz)
        g_feat = self.glob_enc(xyz)
        fused = self.fusion(h_feat, g_feat)
        return self.bottleneck(fused)  # (z_q_st, codebook_loss, indices)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)["recon"]  # extrai só o tensor de pontos

    def forward(self, xyz: torch.Tensor) -> dict:
        z_q, codebook_loss, indices = self.encode(xyz)
        decoder_out = self.decoder(z_q)  # dict completo
        recon = decoder_out["recon"]  # (B, N, 6)
        centers = decoder_out["centers"]  # (B, n_centers, 3) — opcional para debug
        return dict(
            recon=recon,
            z=z_q,
            codebook_loss=codebook_loss,
            indices=indices,
            centers=centers,
        )

    # ── loss ──────────────────────────────────────────────────────────────────
    def loss(self, out: dict, target: torch.Tensor) -> dict:
        cd = chamfer_distance(out["recon"], target)
        # Não há KL — a regularização vem da commitment/codebook loss
        total = cd + out["codebook_loss"]
        return dict(total=total, cd=cd, codebook_loss=out["codebook_loss"])

    # ── geração ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Sorteia n índices aleatórios do codebook e decodifica.
        Para geração condicional, substitua randint por um prior aprendido
        (e.g. PixelCNN / Transformer sobre sequências de índices).
        """
        indices = torch.randint(0, self.n_embeddings, (n,), device=device)
        z = self.bottleneck.decode_indices(indices)  # (n, latent_dim)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self, xyz_a: torch.Tensor, xyz_b: torch.Tensor, steps: int = 8
    ) -> torch.Tensor:
        """
        Interpolação linear no espaço contínuo pré-quantização.
        Cada ponto intermediário é re-quantizado antes de decodificar,
        garantindo que o decoder sempre receba um vetor do codebook.
        """

        # Extrai vetores contínuos antes da quantização
        def _get_ze(xyz):
            h_feat, _ = self.hier_enc(xyz)
            g_feat = self.glob_enc(xyz)
            fused = self.fusion(h_feat, g_feat)
            return self.bottleneck.proj_in(fused)  # (B, latent_dim)

        ze_a = _get_ze(xyz_a)
        ze_b = _get_ze(xyz_b)

        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            ze_interp = (1 - a) * ze_a + a * ze_b
            z_q, _ = self.bottleneck.quantize(ze_interp)  # re-quantiza
            shapes.append(self.decode(z_q))
        return torch.stack(shapes, dim=1)  # (B, steps, N, 3)
