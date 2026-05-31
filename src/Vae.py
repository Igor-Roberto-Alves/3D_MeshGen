import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.Encoder import HierarchicalEncoder, GlobalEncoder, CrossBranchFusion
from src.Decoder import PointDecoder
from src.metric import chamfer_distance
from src.Encoder import VAEBottleneck


class DualBranchPointVAE(nn.Module):
    """
    Full dual-branch VAE estruturado e corrigido para o novo pipeline geométrico.

    Args:
        d_model      : transformer channel dimension
        latent_dim   : size of the VAE latent space
        n_out        : number of output points (reconstruction)
        enc_depth    : transformer layers in each encoder branch
        dec_depth    : transformer layers in decoder
        beta         : β-VAE weight on the KL term
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 512,
        n_out: int = 2048,
        enc_depth: int = 4,
        dec_depth: int = 4,
        n_heads: int = 6,
        beta: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim

        # Shared-weight Siamese encoders
        self.hier_enc = HierarchicalEncoder(
            d_model=d_model, n_heads=n_heads, depth=enc_depth
        )
        self.glob_enc = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion = CrossBranchFusion(d_model=d_model, n_heads=n_heads)
        self.bottleneck = VAEBottleneck(in_dim=d_model, latent_dim=latent_dim)
        
        self.decoder = PointDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=dec_depth,
            n_out=n_out,
        )

    # ── forward ──────────────────────────────────────────────────────────────
    def encode(self, xyz: torch.Tensor):
        h_feat, _ = self.hier_enc(xyz)
        g_feat = self.glob_enc(xyz)
        fused = self.fusion(h_feat, g_feat)
        z, mu, logvar = self.bottleneck(fused)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> dict:
        # Agora o decoder retorna um dicionário {"recon": ..., "centers": ...}
        return self.decoder(z)

    def forward(self, xyz: torch.Tensor) -> dict:
        z, mu, logvar = self.encode(xyz)
        decoder_out = self.decode(z)
        
        # Repassamos o dicionário do decoder e adicionamos os tokens do bottleneck
        return dict(
            recon=decoder_out["recon"], 
            centers=decoder_out["centers"],
            z=z, 
            mu=mu, 
            logvar=logvar
        )

    # ── loss ─────────────────────────────────────────────────────────────────
    def loss(self, out: dict, target: torch.Tensor) -> dict:
        # Puxa o "recon" de dentro do dicionário out que veio do forward
        cd = chamfer_distance(out["recon"], target)
        kl = VAEBottleneck.kl_loss(out["mu"], out["logvar"])
        
        # Defesa contra bugs: garante que se self.beta for 0, o KL zera de verdade no total
        total = 100*cd + (self.beta * kl)
        
        return dict(total=total, cd=cd, kl=kl)

    # ── generation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n point clouds from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        decoder_out = self.decode(z)
        return decoder_out["recon"]

    @torch.no_grad()
    def interpolate(
        self, xyz_a: torch.Tensor, xyz_b: torch.Tensor, steps: int = 8
    ) -> torch.Tensor:
        """Spherical linear interpolation between two shapes."""
        _, mu_a, _ = self.encode(xyz_a)
        _, mu_b, _ = self.encode(xyz_b)
        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            z = (1 - a) * mu_a + a * mu_b
            decoder_out = self.decode(z)
            shapes.append(decoder_out["recon"])
        return torch.stack(shapes, dim=1)  # (B, steps, N, 6)