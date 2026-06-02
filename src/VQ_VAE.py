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
from src.GAN import PointNetDiscriminator

# ── VQ Bottleneck ─────────────────────────────────────────────────────────────


class VQBottleneck(nn.Module):
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

        self.proj_in = nn.Linear(in_dim, latent_dim)

        self.embedding = nn.Embedding(n_embeddings, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1 / n_embeddings, 1 / n_embeddings)

        if ema_update:
            self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
            self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
            self.ema_decay = ema_decay

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = (
            z_e.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * z_e @ self.embedding.weight.T
        )
        indices = d.argmin(1)
        z_q = self.embedding(indices)
        return z_q, indices

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.n_embeddings).float()
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.T @ z_e

            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            self.ema_embed_sum.mul_(self.ema_decay).add_(
                embed_sum, alpha=1 - self.ema_decay
            )

            n = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + 1e-5) / (n + self.n_embeddings * 1e-5) * n
            )
            self.embedding.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.proj_in(h)
        z_q, indices = self.quantize(z_e)

        if self.training and self.ema_update:
            self._ema_update(z_e.detach(), indices)
            codebook_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            codebook_loss = F.mse_loss(
                z_e.detach(), z_q
            ) + self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, codebook_loss, indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    @property
    def codebook(self) -> torch.Tensor:
        return self.embedding.weight


# ── Modelo Principal ──────────────────────────────────────────────────────────


class DualBranchPointVQVAE(nn.Module):
    """
    Dual-branch VQ-VAE + GAN para nuvens de pontos.

    Args:
        d_model         : dimensão interna do transformer
        latent_dim      : dimensão de cada vetor do codebook
        n_embeddings    : número de entradas no codebook
        n_out           : pontos gerados na reconstrução
        enc_depth       : camadas transformer no encoder
        dec_depth       : camadas transformer no decoder
        n_heads         : cabeças de atenção
        commitment_cost : peso da commitment loss (β do VQ-VAE)
        ema_update      : True para atualizar codebook via EMA
        adv_weight      : λ que pondera a adversarial loss no total do gerador
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
        adv_weight: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.adv_weight = adv_weight

        self.hier_enc = HierarchicalEncoder(
            d_model=d_model, n_heads=n_heads, depth=enc_depth
        )
        self.glob_enc = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion = CrossBranchFusion(d_model=d_model, n_heads=n_heads)
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
        self.discriminator = PointNetDiscriminator(in_ch=6, use_spectral=True)

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_feat, _ = self.hier_enc(xyz)
        g_feat = self.glob_enc(xyz)
        fused = self.fusion(h_feat, g_feat)
        return self.bottleneck(fused)  # (z_q_st, codebook_loss, indices)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)["recon"]  # extrai só o tensor (B, N, 6)

    def forward(self, xyz: torch.Tensor) -> dict:
        z_q, codebook_loss, indices = self.encode(xyz)
        decoder_out = self.decoder(z_q)
        recon = decoder_out["recon"]
        centers = decoder_out["centers"]
        return dict(
            recon=recon,
            z=z_q,
            codebook_loss=codebook_loss,
            indices=indices,
            centers=centers,
        )

    # ── loss do gerador (encoder + decoder) ──────────────────────────────────

    def loss(self, out: dict, target: torch.Tensor) -> dict:
        cd = chamfer_distance(out["recon"], target)
        codebook_loss = out["codebook_loss"]

        # Gerador quer enganar o D → labels = 1 para as fakes
        fake_logits = self.discriminator(out["recon"])
        adv = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

        total = cd + codebook_loss + self.adv_weight * adv
        return dict(total=total, cd=cd, codebook_loss=codebook_loss, adv_g=adv)

    # ── loss do discriminador ─────────────────────────────────────────────────

    def discriminator_loss(
        self, recon: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        """
        recon : (B, N, 6)  saída do decoder — .detach() aplicado internamente
        real  : (B, N, 6)  ponto cloud original
        """
        real_logits = self.discriminator(real)
        fake_logits = self.discriminator(recon.detach())  # não treina o gerador aqui

        loss_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        return (loss_real + loss_fake) * 0.5

    # ── geração ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.n_embeddings, (n,), device=device)
        z = self.bottleneck.decode_indices(indices)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(
        self, xyz_a: torch.Tensor, xyz_b: torch.Tensor, steps: int = 8
    ) -> torch.Tensor:
        def _get_ze(xyz):
            h_feat, _ = self.hier_enc(xyz)
            g_feat = self.glob_enc(xyz)
            fused = self.fusion(h_feat, g_feat)
            return self.bottleneck.proj_in(fused)

        ze_a = _get_ze(xyz_a)
        ze_b = _get_ze(xyz_b)

        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            ze_interp = (1 - a) * ze_a + a * ze_b
            z_q, _ = self.bottleneck.quantize(ze_interp)
            shapes.append(self.decode(z_q))
        return torch.stack(shapes, dim=1)  # (B, steps, N, 3)
