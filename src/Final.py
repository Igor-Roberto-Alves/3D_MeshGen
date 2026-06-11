"""
latent_diffusion.py
-------------------
HierarchicalLatentDiffusion
    Wraps an existing Vae (encoder + decoder) with two DDPM priors:
      • StyleDDPM   → learns p(z_style)
      • PointsDDPM  → learns p(z_points | z_style)

Training loss
    L = L_recon  + β · L_KL  + λ_s · L_style_ddpm  + λ_p · L_points_ddpm

Sampling (generation)
    1. Sample z_style  from StyleDDPM   (pure noise → denoised)
    2. Sample z_points from PointsDDPM  conditioned on z_style
    3. Decode(z_style, z_points) → point cloud
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.diffusion.noise_scheduler import DDPMScheduler
from src.diffusion.score_networks  import StyleScoreNet, PointScoreNet


class HierarchicalLatentDiffusion(nn.Module):
    """
    Parameters
    ----------
    vae         : your existing Vae instance (must expose .encoder and .decoder)
    style_dim   : style latent dimension (default: 512)
    latent_dim  : points latent dimension (default: 256)
    T           : diffusion timesteps (default: 1000)
    schedule    : "cosine" | "linear"
    score_depth_style  : ResBlock depth for StyleScoreNet
    score_depth_points : ResBlock depth for PointScoreNet
    lambda_style  : weight for style DDPM loss
    lambda_points : weight for points DDPM loss
    """

    def __init__(
        self,
        vae,
        style_dim:           int   = 512,
        latent_dim:          int   = 256,
        T:                   int   = 1000,
        schedule:            str   = "cosine",
        score_depth_style:   int   = 6,
        score_depth_points:  int   = 8,
        lambda_style:        float = 1.0,
        lambda_points:       float = 1.0,
    ):
        super().__init__()
        self.vae          = vae
        self.style_dim    = style_dim
        self.latent_dim   = latent_dim
        self.lambda_style  = lambda_style
        self.lambda_points = lambda_points

        # ---- Noise schedulers (not nn.Modules — just buffers) ----------
        self.style_scheduler  = DDPMScheduler(T=T, schedule=schedule)
        self.points_scheduler = DDPMScheduler(T=T, schedule=schedule)

        # ---- Score networks -------------------------------------------
        self.style_score_net = StyleScoreNet(
            style_dim=style_dim,
            hidden_dim=max(style_dim, 512),
            depth=score_depth_style,
        )
        self.points_score_net = PointScoreNet(
            latent_dim=latent_dim,
            style_dim=style_dim,
            hidden_dim=max(latent_dim, 512),
            depth=score_depth_points,
        )

    # ================================================================== #
    # Forward pass (training)
    # ================================================================== #
    def forward(
        self,
        points:     torch.Tensor,       # (B, N, 6)
        beta:       float        = 1.0,  # KL weight
        recon_loss: str          = "chamfer",
        emd_weight: float        = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict of all losses. Caller sums them:
            total = losses["recon"] + losses["kl"] + losses["style_ddpm"] + losses["points_ddpm"]
        """
        from src.metric import vae_loss

        # ---- 1. VAE forward: encode ---------------------------------- #
        coords_pred, normals_pred, mu, logvar = self.vae(points)

        # Extract the two latent codes from the VAE's encoder output.
        # Assumption: your Vae stores mu_style / mu_points after encode.
        # If not, split mu by dimension:
        #   mu_style  = mu[:, :self.style_dim]
        #   mu_points = mu[:, self.style_dim:]
        # Adjust the split below to match your Vae's actual layout.
        z_style, z_points = self._split_latents(mu, logvar)

        target_xyz = points[..., :3]

        # ---- 2. VAE reconstruction + KL loss ------------------------- #
        vae_losses = vae_loss(
            pred_xyz   = coords_pred,
            target_xyz = target_xyz,
            mu         = mu,
            logvar     = logvar,
            beta       = beta,
            recon_loss = recon_loss,
            emd_weight = emd_weight,
        )

        # ---- 3. Style DDPM loss -------------------------------------- #
        # score network wrapped as a callable matching DDPMScheduler API
        def style_model(z_t, t):
            return self.style_score_net(z_t, t)

        loss_style = self.style_scheduler.training_loss(
            model = style_model,
            x0    = z_style.detach(),   # detach: diffusion prior is trained
        )                               # separately from the VAE encoder

        # ---- 4. Points DDPM loss (conditioned on z_style) ----------- #
        def points_model(z_t, t, z_style_cond):
            return self.points_score_net(z_t, t, z_style_cond)

        loss_points = self.points_scheduler.training_loss(
            model          = points_model,
            x0             = z_points.detach(),
            z_style_cond   = z_style.detach(),
        )

        return {
            "recon":        vae_losses["recon"],
            "kl":           vae_losses["kl"],
            "total_vae":    vae_losses["total"],
            "style_ddpm":   loss_style  * self.lambda_style,
            "points_ddpm":  loss_points * self.lambda_points,
            "total": (
                vae_losses["total"]
                + loss_style  * self.lambda_style
                + loss_points * self.lambda_points
            ),
        }

    # ================================================================== #
    # Generation (sampling from the learned prior)
    # ================================================================== #
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        num_points:  int,
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Generates point clouds from the hierarchical prior.
        1. z_style  ← StyleDDPM.sample()
        2. z_points ← PointsDDPM.sample(conditioned on z_style)
        3. Decode   → (B, num_points, 3)
        """
        # Step 1: sample style
        z_style = self.style_scheduler.sample(
            model  = lambda z, t: self.style_score_net(z, t),
            shape  = (num_samples, self.style_dim),
            device = device,
        )

        # Step 2: sample points conditioned on style
        z_points = self.points_scheduler.sample(
            model          = lambda z, t, **kw: self.points_score_net(z, t, kw["z_style_cond"]),
            shape          = (num_samples, self.latent_dim),
            device         = device,
            z_style_cond   = z_style,
        )

        # Step 3: decode
        # Your decoder likely expects (z_style, z_points) or a combined z.
        # Adjust this call to match your Vae.decoder interface:
        xyz, _normals = self.vae.decoder(z_style, z_points, num_points=num_points)
        return xyz

    # ================================================================== #
    # Helpers
    # ================================================================== #
    def _split_latents(
        self,
        mu:     torch.Tensor,   # (B, style_dim + latent_dim)  or structured
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the combined (mu, logvar) into z_style and z_points using the
        reparametrization trick independently on each branch.

        If your Vae already returns them separately (e.g. mu_style, mu_points),
        replace this method to use those directly instead.
        """
        eps_style  = torch.randn_like(mu[:, :self.style_dim])
        eps_points = torch.randn_like(mu[:, self.style_dim:])

        mu_style   = mu[:, :self.style_dim]
        mu_points  = mu[:, self.style_dim:]
        lv_style   = logvar[:, :self.style_dim]
        lv_points  = logvar[:, self.style_dim:]

        z_style  = mu_style  + eps_style  * (0.5 * lv_style).exp()
        z_points = mu_points + eps_points * (0.5 * lv_points).exp()
        return z_style, z_points

    # ================================================================== #
    # Separate parameter groups for optimizers
    # ================================================================== #
    def vae_parameters(self):
        return self.vae.parameters()

    def diffusion_parameters(self):
        return list(self.style_score_net.parameters()) + list(self.points_score_net.parameters())