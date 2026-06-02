import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.Encoder import HierarchicalEncoder, GlobalEncoder, CrossBranchFusion, VAEBottleneck
from src.Decoder import PointDecoder
from src.GAN import PointNetDiscriminator
from src.metric import chamfer_distance


class DualBranchPointVAE(nn.Module):
    """
    Dual-branch β-VAE + GAN for point cloud generation.

    Training has two alternating phases (handled in train_vae_gan.py):

        Phase 1 — Discriminator step
            • real batch  → D → real_loss  (label = 1)
            • recon batch → D → fake_loss  (label = 0)
            • optimiser_D.step()

        Phase 2 — Generator (VAE) step
            • reconstruction loss  : Chamfer distance
            • KL divergence        : β-weighted
            • adversarial loss     : fool D  (label = 1 on fake)
            • optimiser_G.step()

    The discriminator is a member of this class so checkpointing is simple,
    but its parameters are frozen during the generator step via the helper
    methods  freeze_D / unfreeze_D.

    Args:
        d_model      : transformer channel dimension
        latent_dim   : VAE latent space size
        n_out        : output points per cloud
        enc_depth    : transformer layers in each encoder branch
        dec_depth    : transformer layers in decoder
        n_heads      : attention heads (encoder + decoder)
        beta         : β weight on the KL term
        lambda_adv   : weight on the adversarial loss in the G step
                       set to 0.0 to disable GAN and train as plain β-VAE
        disc_base_ch : base channel width of the discriminator MLP
    """

    def __init__(
        self,
        d_model: int = 384,
        latent_dim: int = 2048 * 6,
        n_out: int = 2048,
        enc_depth: int = 4,
        dec_depth: int = 4,
        n_heads: int = 6,
        beta: float = 1e-3,
        lambda_adv: float = 0.1,
        disc_base_ch: int = 64,
    ):
        super().__init__()
        self.beta       = beta
        self.lambda_adv = lambda_adv
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        self.hier_enc   = HierarchicalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.glob_enc   = GlobalEncoder(d_model=d_model, n_heads=n_heads, depth=enc_depth)
        self.fusion     = CrossBranchFusion(d_model=d_model, n_heads=n_heads)
        self.bottleneck = VAEBottleneck(in_dim=d_model, latent_dim=latent_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = PointDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=dec_depth,
            n_out=n_out,
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

    def encode(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_feat, _ = self.hier_enc(xyz)
        g_feat    = self.glob_enc(xyz)
        fused     = self.fusion(h_feat, g_feat)
        z, mu, logvar = self.bottleneck(fused)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> dict:
        return self.decoder(z)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, xyz: torch.Tensor) -> dict:
        z, mu, logvar = self.encode(xyz)
        decoder_out   = self.decode(z)
        return dict(
            recon   = decoder_out["recon"],
            z       = z,
            mu      = mu,
            logvar  = logvar,
        )

    # ── losses ────────────────────────────────────────────────────────────────

    def loss_discriminator(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> dict:
        """
        Compute the discriminator loss on one batch.

        Call BEFORE loss_generator. Do NOT call freeze_D here — the
        discriminator must be unfrozen to accumulate gradients.

        real : (B, N, 6)  ground-truth point cloud
        fake : (B, N, 6)  reconstructed / generated cloud  (.detach() applied
                           inside so G gradients do not flow here)
        """
        B = real.size(0)
        device = real.device

        real_labels = torch.ones(B,  1, device=device)
        fake_labels = torch.zeros(B, 1, device=device)

        loss_real = self.adv_loss(self.discriminator(real),         real_labels)
        loss_fake = self.adv_loss(self.discriminator(fake.detach()), fake_labels)

        loss_D = (loss_real + loss_fake) * 0.5
        return dict(loss_D=loss_D, loss_D_real=loss_real, loss_D_fake=loss_fake)

    def loss_generator(
        self,
        out: dict,
        target: torch.Tensor,
    ) -> dict:
        """
        Compute the full VAE + adversarial generator loss.

        Freeze the discriminator with freeze_D() before calling this so D
        parameters do not receive gradients during the G step.

        out    : dict returned by forward()
        target : (B, N, 6)  ground-truth point cloud
        """
        B      = out["recon"].size(0)
        device = out["recon"].device

        # Reconstruction (Chamfer) + KL
        cd = chamfer_distance(out["recon"], target)
        kl = VAEBottleneck.kl_loss(out["mu"], out["logvar"])

        # Adversarial: fool D into predicting fake as real
        real_labels = torch.ones(B, 1, device=device)
        loss_adv = self.adv_loss(self.discriminator(out["recon"]), real_labels)

        total = 100.0 * cd + self.beta * kl + self.lambda_adv * loss_adv

        return dict(total=total, cd=cd, kl=kl, loss_adv=loss_adv)

    # ── kept for backward-compat with non-GAN training loops ─────────────────

    def loss(self, out: dict, target: torch.Tensor) -> dict:
        """Plain β-VAE loss (no adversarial term). Useful for warm-up epochs."""
        cd    = chamfer_distance(out["recon"], target)
        kl    = VAEBottleneck.kl_loss(out["mu"], out["logvar"])
        total = 100.0 * cd + self.beta * kl
        return dict(total=total, cd=cd, kl=kl)

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n point clouds from the prior N(0, I)."""
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
        """Linear or spherical interpolation between two shapes."""
        _, mu_a, _ = self.encode(xyz_a)
        _, mu_b, _ = self.encode(xyz_b)
        alphas = torch.linspace(0, 1, steps, device=xyz_a.device)
        shapes = []
        for a in alphas:
            if use_slerp:
                dot       = (mu_a * mu_b).sum(dim=-1, keepdim=True).clamp(-1, 1)
                omega     = torch.acos(dot)
                sin_omega = torch.sin(omega).clamp(min=1e-8)
                z = (torch.sin((1 - a) * omega) / sin_omega) * mu_a \
                  + (torch.sin(a * omega)        / sin_omega) * mu_b
            else:
                z = (1 - a) * mu_a + a * mu_b
            shapes.append(self.decode(z)["recon"])
        return torch.stack(shapes, dim=1)   # (B, steps, N, 6)