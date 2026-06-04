import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.Encoder import (
    HierarchicalEncoder,
    GlobalEncoder,
    CrossBranchFusion,
    VAEBottleneck,
)

# Keep your original GAN and loss modules
from src.GAN import PointNetDiscriminator
from src.metric import chamfer_distance, earth_movers_distance_sinkhorn


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.LionDecoder import *



class DualBranchPointVAE(nn.Module):
    """
    Dual-branch β-VAE + GAN for point cloud generation with MiniLion Decoder.
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 512,
        n_out: int = 2048,
        enc_depth: int = 4,
        dec_depth: int = 4,  # Kept for signature backward compatibility
        n_heads: int = 6,    # Kept for signature backward compatibility
        beta: float = 1e-3,
        lambda_adv: float = 0.1,
        disc_base_ch: int = 64,
    ):
        super().__init__()
        self.beta = beta
        self.lambda_adv = lambda_adv
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        self.hier_enc = HierarchicalEncoder(
            d_model=d_model, n_heads=n_heads, depth=enc_depth
        )
        self.glob_enc = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion = CrossBranchFusion(d_model=d_model, n_heads=n_heads)
        self.bottleneck = VAEBottleneck(in_dim=d_model, latent_dim=latent_dim)

        # ── INTEGRATED: New LION Style/AdaGN-based Decoder ────────────────────
        self.decoder = NvidiaStyleLatentDecoder(
            latent_dim=latent_dim,
            style_dim=d_model,
            n_out=n_out,
            n_seed=256,  # Can be tuned for quality/speed trade-off
        )

        # ── Discriminator ─────────────────────────────────────────────────────
        self.discriminator = PointNetDiscriminator(
            in_ch=6,
            base_ch=disc_base_ch,
            use_spectral=True,
        )

        # BCE loss used for both D and G adversarial steps
        self.adv_loss = nn.BCEWithLogitsLoss()

    # ── helpers ───────────────────────────────────────────────────────────────

    def freeze_D(self) -> None:
        """Freeze discriminator weights (call before the G/VAE update step)."""
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

    def unfreeze_D(self) -> None:
        """Unfreeze discriminator weights (call before the D update step)."""
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_feat, _ = self.hier_enc(xyz)
        g_feat = self.glob_enc(xyz)  # <--- Este é o seu STYLE original (B, d_model)
        
        fused = self.fusion(h_feat, g_feat)
        z, mu, logvar = self.bottleneck(fused)
        
        return z, mu, logvar, g_feat  # <--- Passamos o g_feat para fora

    def decode(self, z: torch.Tensor, style: torch.Tensor) -> dict:
        # O novo decoder precisa receber tanto o z (pontos latentes) quanto o style
        return self.decoder(z_local=z, style=style)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, xyz: torch.Tensor) -> dict:
        z, mu, logvar, style = self.encode(xyz) # <--- Captura o style aqui
        decoder_out = self.decode(z, style)     # <--- Passa o style para o decode
        return dict(
            recon=decoder_out["recon"],
            z=z,
            mu=mu,
            logvar=logvar,
        )
    # ── losses ────────────────────────────────────────────────────────────────

    def loss_discriminator(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> dict:
        # Usando Hinge Loss para o Discriminador
        d_real_logits = self.discriminator(real)
        d_fake_logits = self.discriminator(fake.detach())

        loss_real = F.relu(1.0 - d_real_logits).mean()
        loss_fake = F.relu(1.0 + d_fake_logits).mean()

        loss_D = loss_real + loss_fake
        return dict(loss_D=loss_D, loss_D_real=loss_real, loss_D_fake=loss_fake)

    def loss_generator(
        self,
        out: dict,
        target: torch.Tensor,
    ) -> dict:
        B = out["recon"].size(0)
        
        point_loss, normal_loss = chamfer_distance(out["recon"], target)
        kl = VAEBottleneck.kl_loss(out["mu"], out["logvar"])

        # Hinge Loss para o Gerador
        g_fake_logits = self.discriminator(out["recon"])
        loss_adv = -g_fake_logits.mean()

        total = point_loss + self.beta * kl + self.lambda_adv * loss_adv + normal_loss

        return dict(
            total=total,
            cd=point_loss,
            kl=kl,
            loss_adv=loss_adv,
            normal_loss=normal_loss,
        )

    def loss(self, out: dict, target: torch.Tensor) -> dict:
        """Plain β-VAE loss (no adversarial term). Useful for warm-up epochs."""
        point_loss, normal_loss = chamfer_distance(out["recon"], target)
        kl = VAEBottleneck.kl_loss(out["mu"], out["logvar"])
        total = point_loss + self.beta * kl + normal_loss
        return dict(total=total, cd=point_loss, kl=kl, normal_loss=normal_loss)

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)["recon"]

    @torch.no_grad()
    def interpolate(
        self,
        xyz_a: torch.Tensor,
        xyz_b: torch.Tensor,
        steps: int = 8,
        use_slerp: bool = False,
    ) -> torch.Tensor:
        _, mu_a, _ = self.encode(xyz_a)
        _, mu_b, _ = self.encode(xyz_b)
        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            if use_slerp:
                dot = (mu_a * mu_b).sum(dim=-1, keepdim=True).clamp(-1, 1)
                omega = torch.acos(dot)
                sin_omega = torch.sin(omega).clamp(min=1e-8)
                z = (torch.sin((1 - a) * omega) / sin_omega) * mu_a + (
                    torch.sin(a * omega) / sin_omega
                ) * mu_b
            else:
                z = (1 - a) * mu_a + a * mu_b
            shapes.append(self.decode(z)["recon"])
        return torch.stack(shapes, dim=1)

    def report_parameters(self) -> None:
        def count_module_params(module: nn.Module):
            tot = sum(p.numel() for p in module.parameters())
            train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return tot, train

        h_tot, h_train = count_module_params(self.hier_enc)
        g_tot, g_train = count_module_params(self.glob_enc)
        f_tot, f_train = count_module_params(self.fusion)
        b_tot, b_train = count_module_params(self.bottleneck)
        dec_tot, dec_train = count_module_params(self.decoder)
        disc_tot, disc_train = count_module_params(self.discriminator)

        global_tot, global_train = count_module_params(self)

        print("\n" + "=" * 65)
        print(f"{'SUBMODULE':<25} | {'TOTAL PARAMS':<16} | {'TRAINABLE PARAMS':<16}")
        print("=" * 65)
        print(f"{'Hierarchical Encoder':<25} | {h_tot:>14,} | {h_train:>16,}")
        print(f"{'Global Encoder':<25} | {g_tot:>14,} | {g_train:>16,}")
        print(f"{'Cross-Branch Fusion':<25} | {f_tot:>14,} | {f_train:>16,}")
        print(f"{'VAE Bottleneck':<25} | {b_tot:>14,} | {b_train:>16,}")
        print(f"{'Point Decoder (LION)':<25} | {dec_tot:>14,} | {dec_train:>16,}")
        print(f"{'Discriminator (PointNet)':<25} | {disc_tot:>14,} | {disc_train:>16,}")
        print("-" * 65)
        print(f"{'GLOBAL SYSTEM TOTAL':<25} | {global_tot:>14,} | {global_train:>16,}")
        print("=" * 65 + "\n")