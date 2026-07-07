"""
train_diffusion.py
------------------
Train the two LION diffusion stages on top of a frozen pretrained VAE.

  Stage 1  StyleDenoiser       :  class label → z_g          (global style DDPM)
  Stage 2  LatentPointDenoiser :  z_g         → z_local      (latent point DDPM)

Both are trained simultaneously.  Each batch:
  1. Run the frozen VAE encoder → get CLEAN latents (posterior mean, no noise).
  2. Sample random timesteps → add noise via the cosine schedule.
  3. Ask each denoiser to predict the noise → MSE loss.
  4. Backprop only through the denoisers (VAE is frozen).

Generation (end of training / TensorBoard):
  1. Sample z_g  from StyleDenoiser prior (class-conditioned, DDPM chain).
  2. Sample z_local from LatentPointDenoiser prior (z_g-conditioned, DDPM chain).
  3. Decode z_local via the frozen VAE decoder.
"""

import argparse
import json
import logging
from contextlib import contextmanager
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import Ds_point_sampled_already
from src.Vae import Vae, normalize_pc
from src.vae_up import VaeUp
from src.Diffusion import CosineSchedule, StyleDenoiser, LatentPointDenoiser
from src.metric import generative_metrics


# ============================================================
# Configuration
# ============================================================

@dataclass
class DiffusionConfig:
    # --- paths ---
    vae_ckpt:    str   = "checkpoints/best.pt"
    ckpt_dir:    str   = "checkpoints_diff"
    log_dir:     str   = "logs_diff"
    data_root:   str   = "point_clouds"

    # --- VAE architecture (must match the loaded checkpoint) ---
    vae_type:        str = "up"  # "up" → VaeUp (fold decoder), "base" → Vae
    vae_latent_dim:  int = 8    # must match the latent_dim used in train_vae.py
    vae_n_latent:    int = 512  # coarse latent points — only used when vae_type="up"
    vae_style_dim:   int = 256
    vae_in_channels: int = 6

    # --- diffusion schedule ---
    T:           int   = 1000

    # --- Style denoiser (Stage 1) ---
    num_classes:    int   = 55
    style_hidden:   int   = 512
    style_layers:   int   = 6
    cfg_dropout:    float = 0.1
    guidance:       float = 3.0    # CFG scale at inference

    # --- Latent point denoiser (Stage 2) — DiT over the N latent points ---
    point_hidden:    int   = 256
    point_layers:    int   = 8
    point_heads:     int   = 8      # attention heads per DiT block (must divide point_hidden)
    point_mlp_ratio: float = 4.0    # FFN expansion inside each DiT block

    # --- training ---
    epochs:         int   = 300
    batch_size:     int   = 16
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0
    min_snr_gamma:  float = 5.0    # Min-SNR-γ loss weighting (<=0 desativa)
    ema_decay:      float = 0.999  # EMA dos pesos p/ amostragem (<=0 desativa)

    # --- data ---
    val_split:   float = 0.1
    test_split:  float = 0.5   # fraction of val_pool that becomes the test set
    num_workers: int   = 4
    pin_memory:  bool  = True

    # --- misc ---
    seed:        int   = 42
    save_every:  int   = 10
    log_every:   int   = 50
    device:      str   = "cuda"
    amp:         bool  = True
    resume:      int   = 0

    # --- generation visualization ---
    vis_every:      int   = 10    # log generated meshes every N epochs
    vis_per_class:  int   = 2     # samples per class to visualize

    # --- generation metrics (MMD-CD / COV-CD / 1-NNA-CD) ---
    eval_every:  int = 50     # compute every N epochs — expensive (full DDPM sampling)
    eval_n_gen:  int = 16     # generated samples per class
    eval_n_ref:  int = 16     # held-out real samples per class to compare against
    eval_points: int = 1024   # subsample point clouds to this many points before CD
    eval_chunk:  int = 4      # row-chunk size for the pairwise CD matrix (memory control)


# ============================================================
# Latent extraction (frozen VAE)
# ============================================================

@torch.no_grad()
def extract_latents(vae: Vae | VaeUp, points: torch.Tensor):
    """
    Run the frozen VAE encoder and return CLEAN latents (posterior mean, no sampling).

    Using the mean rather than a sample is the standard approach when training
    a latent diffusion model (Stable Diffusion does the same).  It gives the
    diffusion a deterministic, noise-free target and makes training more stable.

    Returns
    -------
    z_g  (B, style_dim)       global style mean
    z_l  (B, N, latent_dim)   local posterior mean — no anchor prefix
    """
    x_norm  = normalize_pc(points)
    mu_g, _ = vae.global_encoder(x_norm)   # (B, style_dim)
    mu_l, _ = vae.local_encoder(x_norm, mu_g)  # (B, N, latent_dim)
    return mu_g, mu_l


class LatentNormalizer:
    """
    Per-dimension affine normalisation of VAE latents.

    The cosine DDPM schedule is calibrated for N(0,I) data; VAE posterior
    means typically have std << 1 (e.g. 0.44 for z_g, 0.70 for z_l).
    This mismatch shifts the effective SNR at every timestep and produces a
    floor in loss_point that can't be crossed regardless of capacity.

    We compute per-dim mean and std once over the training set, normalise
    both latents to zero mean / unit variance before diffusion, and
    denormalise after sampling for the VAE decoder.
    """

    def __init__(self,
                 zg_mean: torch.Tensor, zg_std: torch.Tensor,
                 zl_mean: torch.Tensor, zl_std: torch.Tensor):
        self.zg_mean = zg_mean   # (style_dim,)
        self.zg_std  = zg_std    # (style_dim,)
        self.zl_mean = zl_mean   # (latent_dim,)
        self.zl_std  = zl_std    # (latent_dim,)

    def norm_zg(self, z_g: torch.Tensor) -> torch.Tensor:
        return (z_g - self.zg_mean) / self.zg_std

    def norm_zl(self, z_l: torch.Tensor) -> torch.Tensor:
        return (z_l - self.zl_mean) / self.zl_std

    def denorm_zg(self, z_g_n: torch.Tensor) -> torch.Tensor:
        return z_g_n * self.zg_std + self.zg_mean

    def denorm_zl(self, z_l_n: torch.Tensor) -> torch.Tensor:
        return z_l_n * self.zl_std + self.zl_mean

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in
                ("zg_mean", "zg_std", "zl_mean", "zl_std")}

    @classmethod
    def from_state_dict(cls, d: dict, device):
        return cls(**{k: d[k].to(device) for k in
                      ("zg_mean", "zg_std", "zl_mean", "zl_std")})


@torch.no_grad()
def compute_latent_stats(
    vae:    Vae | VaeUp,
    loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
) -> LatentNormalizer:
    """One pass over the training set to get per-dim mean/std of z_g and z_l."""
    logger.info("Computing latent statistics for normalisation (one pass)...")

    zg_sum = zg_sq = zg_n = 0
    zl_sum = zl_sq = zl_n = 0

    for points, _ in loader:
        points = points.to(device)
        z_g, z_l = extract_latents(vae, points)      # (B, S), (B, N, L)
        B = z_g.shape[0]

        zg_sum += z_g.sum(0);  zg_sq += (z_g ** 2).sum(0);  zg_n += B

        zl_flat = z_l.reshape(-1, z_l.shape[-1])     # (B*N, L)
        zl_sum += zl_flat.sum(0);  zl_sq += (zl_flat ** 2).sum(0);  zl_n += zl_flat.shape[0]

    zg_mean = zg_sum / zg_n
    zg_std  = ((zg_sq / zg_n) - zg_mean ** 2).clamp(min=1e-6).sqrt()
    zl_mean = zl_sum / zl_n
    zl_std  = ((zl_sq / zl_n) - zl_mean ** 2).clamp(min=1e-6).sqrt()

    logger.info(
        f"  z_g  mean_abs={zg_mean.abs().mean():.4f}  "
        f"std [min={zg_std.min():.4f}  mean={zg_std.mean():.4f}  max={zg_std.max():.4f}]"
    )
    logger.info(
        f"  z_l  mean_abs={zl_mean.abs().mean():.4f}  "
        f"std [min={zl_std.min():.4f}  mean={zl_std.mean():.4f}  max={zl_std.max():.4f}]"
    )
    return LatentNormalizer(zg_mean, zg_std, zl_mean, zl_std)


# ============================================================
# Latent caching (precompute once, train on the cache)
# ============================================================

@torch.no_grad()
def build_latent_cache(
    vae:    Vae | VaeUp,
    subset,
    cfg:    "DiffusionConfig",
    device: torch.device,
    logger: logging.Logger,
    tag:    str,
) -> torch.utils.data.TensorDataset:
    """
    Precompute (z_g, z_l, class_idx) for every sample once, using the frozen VAE
    with augmentation OFF and the (now deterministic) FPS.

    This removes the VAE encoder — the dominant per-step cost — from the training
    loop entirely, so many more epochs fit in the same wall-clock budget, and it
    makes the diffusion target fully deterministic (the same shape always yields
    the same z_l), which is what we want for stable convergence.
    """
    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    ZG, ZL, CLS = [], [], []
    for points, cls in loader:
        z_g, z_l = extract_latents(vae, points.to(device))
        ZG.append(z_g.cpu()); ZL.append(z_l.cpu()); CLS.append(cls)
    ZG, ZL, CLS = torch.cat(ZG), torch.cat(ZL), torch.cat(CLS).long()
    logger.info(f"  cached {tag}: z_g {tuple(ZG.shape)}  z_l {tuple(ZL.shape)}  ({CLS.numel()} samples)")
    return torch.utils.data.TensorDataset(ZG, ZL, CLS)


def normalizer_from_cache(cache, device, logger) -> LatentNormalizer:
    """Per-dim mean/std of the cached latents (replaces the extra stat pass)."""
    ZG, ZL, _ = cache.tensors
    zg_mean, zg_std = ZG.mean(0), ZG.std(0).clamp(min=1e-6)
    zl_flat = ZL.reshape(-1, ZL.shape[-1])
    zl_mean, zl_std = zl_flat.mean(0), zl_flat.std(0).clamp(min=1e-6)
    logger.info(
        f"Latent stats (from cache): "
        f"z_g std mean={zg_std.mean():.4f}  z_l std mean={zl_std.mean():.4f}"
    )
    return LatentNormalizer(zg_mean.to(device), zg_std.to(device),
                            zl_mean.to(device), zl_std.to(device))


# ============================================================
# EMA (exponential moving average of the denoiser weights)
# ============================================================

class EMA:
    """
    Mantém uma cópia suavizada dos pesos, usada para AMOSTRAR (não para treinar).

    Difusão treina com loss duplamente estocástica (t e ε aleatórios por step),
    então os pesos balançam mesmo já convergidos. Amostrar da média EMA em vez
    dos pesos crus dá amostras bem mais limpas — prática padrão (DDPM/LION/EDM).

    O decay tem warmup adaptativo `min(decay, (1+step)/(10+step))` para não ficar
    travado no início quando o treino é curto (poucos steps).
    """
    def __init__(self, model, decay: float = 0.999):
        self.decay  = decay
        self.step   = 0
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(d).add_(p.detach(), alpha=1 - d)

    def copy_to(self, model):
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.data.copy_(self.shadow[k])

    def store(self, model):
        self._backup = {k: p.detach().clone()
                        for k, p in model.named_parameters() if p.requires_grad}

    def restore(self, model):
        for k, p in model.named_parameters():
            if k in self._backup:
                p.data.copy_(self._backup[k])
        self._backup = None

    def state_dict(self):
        return {"decay": self.decay, "step": self.step,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd, device):
        self.decay  = sd["decay"]
        self.step   = sd["step"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}


@contextmanager
def use_ema(pairs):
    """Troca temporariamente os pesos dos modelos pelos EMA; restaura ao sair.
    pairs: lista de (ema, model)."""
    for ema, m in pairs:
        ema.store(m); ema.copy_to(m)
    try:
        yield
    finally:
        for ema, m in pairs:
            ema.restore(m)


# ============================================================
# Loss
# ============================================================

def diffusion_loss(
    schedule:      CosineSchedule,
    denoiser:      nn.Module,
    x0:            torch.Tensor,
    condition:     torch.Tensor,
    min_snr_gamma: float | None = 5.0,
) -> torch.Tensor:
    """
    DDPM ε-prediction MSE loss with optional Min-SNR-γ weighting (Hang et al. 2023).

    With uniform-t sampling the many easy low-noise steps dominate the gradient
    and slow convergence. Min-SNR truncates each timestep's weight to
    min(SNR_t, γ)/SNR_t (the ε-prediction form), which down-weights the low-noise
    steps and lets the model spend capacity where signal actually exists. γ=5 is
    the value from the paper; pass None to disable.
    """
    B  = x0.shape[0]
    t  = torch.randint(0, schedule.T, (B,), device=x0.device)
    xt, noise = schedule.q_sample(x0, t)
    pred      = denoiser(xt, t, condition)

    mse = F.mse_loss(pred, noise, reduction="none").flatten(1).mean(1)   # (B,)
    if min_snr_gamma is not None:
        snr    = schedule.acp[t] / (1 - schedule.acp[t]).clamp(min=1e-8)  # (B,)
        weight = snr.clamp(max=min_snr_gamma) / snr.clamp(min=1e-8)
        mse    = weight * mse
    return mse.mean()


# ============================================================
# TensorBoard helpers
# ============================================================

def log_metrics(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


def _side_by_side(clouds, colors, gap=2.5):
    shifted_v, shifted_c = [], []
    x_cursor = 0.0
    for v, c in zip(clouds, colors):
        v = v.clone()
        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += x_cursor
        x_cursor += (v[:, 0].max() - v[:, 0].min()).item() + gap
        shifted_v.append(v)
        shifted_c.append(c)
    return (torch.cat(shifted_v).unsqueeze(0),
            torch.cat(shifted_c).unsqueeze(0))


N_POINTS = 2048   # points per cloud — matches Ds_point_sampled_already


@torch.no_grad()
def sample_class(
    schedule:   CosineSchedule,
    style_dn:   StyleDenoiser,
    point_dn:   LatentPointDenoiser,
    vae:        Vae | VaeUp,
    cfg:        DiffusionConfig,
    cls_idx:    int,
    B:          int,
    device:     torch.device,
    normalizer: LatentNormalizer | None = None,
) -> torch.Tensor:
    """Full two-stage DDPM chain for one class → decoded (B, N_POINTS, 3) point clouds."""
    cls = torch.full((B,), cls_idx, device=device, dtype=torch.long)

    # --- Stage 1: sample normalised z_g ---
    uncond = style_dn.uncond(B, device)
    z_g_n  = schedule.sample(
        style_dn, (cfg.vae_style_dim,),
        condition=cls, uncond=uncond,
        guidance=cfg.guidance, device=device,
    )   # (B, style_dim) — normalised space

    # --- Stage 2: sample normalised z_local conditioned on normalised z_g ---
    n_lat   = getattr(vae, "n_latent", N_POINTS)   # VaeUp → 512, Vae → 2048
    z_l_n   = schedule.sample(
        point_dn, (n_lat, cfg.vae_latent_dim),
        condition=z_g_n, device=device,
    )   # (B, n_lat, latent_dim) — normalised space

    # --- Denormalise before decoding ---
    if normalizer is not None:
        z_g     = normalizer.denorm_zg(z_g_n)
        z_local = normalizer.denorm_zl(z_l_n)
    else:
        z_g, z_local = z_g_n, z_l_n

    return vae.decoder(z_local, z_g).float()  # (B, N_POINTS, 3)


@torch.no_grad()
def log_generations(
    writer:          SummaryWriter,
    schedule:        CosineSchedule,
    style_dn:        StyleDenoiser,
    point_dn:        LatentPointDenoiser,
    vae:             Vae | VaeUp,
    cfg:             DiffusionConfig,
    epoch:           int,
    device:          torch.device,
    class_names:     dict,          # idx → name string
    present_classes: list,          # class indices that actually exist in the dataset
    normalizer:      LatentNormalizer | None = None,
):
    """
    For a handful of classes, generate shapes via the full two-stage chain
    and log them to TensorBoard alongside a real sample from that class.
    """
    style_dn.eval(); point_dn.eval(); vae.eval()

    show_classes = present_classes[:4]

    for cls_idx in show_classes:
        B = cfg.vis_per_class
        xyz_out = sample_class(
            schedule, style_dn, point_dn, vae, cfg, cls_idx, B, device, normalizer
        ).cpu()

        for i in range(B):
            gen_v = xyz_out[i]
            gen_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(N_POINTS, -1)
            verts, clrs = _side_by_side([gen_v], [gen_c])
            cls_name    = class_names.get(cls_idx, str(cls_idx))
            writer.add_mesh(
                f"generated/{cls_name}/sample{i}",
                vertices=verts, colors=clrs, global_step=epoch,
            )

    style_dn.train(); point_dn.train()


@torch.no_grad()
def evaluate_generation_metrics(
    writer:          SummaryWriter,
    schedule:        CosineSchedule,
    style_dn:        StyleDenoiser,
    point_dn:        LatentPointDenoiser,
    vae:             Vae | VaeUp,
    cfg:             DiffusionConfig,
    epoch:           int,
    device:          torch.device,
    class_names:     dict,                    # idx → name string
    ref_by_class:    dict,                    # idx → (M, N_POINTS, 3) real xyz, normalised
    logger:          logging.Logger,
    normalizer:      LatentNormalizer | None = None,
) -> dict:
    """
    MMD-CD / COV-CD / 1-NNA-CD against held-out real samples, per class.

    diffusion_loss (train/val) only measures whether the network denoises
    individual noised examples well — it says nothing about whether the
    *distribution* it generates matches the real one (mode collapse and
    off-distribution shapes can both hide behind a low denoising loss).
    These are the standard metrics (PointFlow / LION) for picking a
    checkpoint by actual generation quality. Expensive: full 1000-step DDPM
    sampling per class, so only run every cfg.eval_every epochs.
    """
    style_dn.eval(); point_dn.eval(); vae.eval()
    agg = {"mmd": [], "cov": [], "nna_1nn": []}

    for cls_idx, ref_xyz in ref_by_class.items():
        n_ref = min(cfg.eval_n_ref, ref_xyz.shape[0])
        if n_ref < 2:
            continue
        ref_idx = torch.randperm(ref_xyz.shape[0], device=device)[:n_ref]
        ref = ref_xyz[ref_idx]

        gen = sample_class(schedule, style_dn, point_dn, vae, cfg, cls_idx, cfg.eval_n_gen, device, normalizer)

        if cfg.eval_points < gen.shape[1]:
            g_idx = torch.randperm(gen.shape[1], device=device)[:cfg.eval_points]
            r_idx = torch.randperm(ref.shape[1], device=device)[:cfg.eval_points]
            gen, ref = gen[:, g_idx], ref[:, r_idx]

        m = generative_metrics(gen, ref, chunk=cfg.eval_chunk)
        name = class_names.get(cls_idx, str(cls_idx))
        writer.add_scalar(f"eval_mmd/{name}",  m["mmd"],     epoch)
        writer.add_scalar(f"eval_cov/{name}",  m["cov"],     epoch)
        writer.add_scalar(f"eval_1nna/{name}", m["nna_1nn"], epoch)
        for k in agg:
            agg[k].append(m[k].item())

    style_dn.train(); point_dn.train()

    if not agg["mmd"]:
        return {}

    summary = {k: sum(v) / len(v) for k, v in agg.items()}
    writer.add_scalar("eval/mmd_mean",  summary["mmd"],     epoch)
    writer.add_scalar("eval/cov_mean",  summary["cov"],     epoch)
    writer.add_scalar("eval/1nna_mean", summary["nna_1nn"], epoch)
    logger.info(
        f"  [eval @ epoch {epoch:03d}]  "
        f"MMD-CD={summary['mmd']:.5f}  COV-CD={summary['cov']:.3f}  "
        f"1-NNA-CD={summary['nna_1nn']:.3f}  (0.5=ideal, 1.0=fully separable)"
    )
    return summary


# ============================================================
# Train / Val steps
# ============================================================

def train_one_epoch(
    schedule:   CosineSchedule,
    style_dn:   StyleDenoiser,
    point_dn:   LatentPointDenoiser,
    loader:     DataLoader,
    opt_s:      torch.optim.Optimizer,
    opt_p:      torch.optim.Optimizer,
    scaler_s:   torch.amp.GradScaler,
    scaler_p:   torch.amp.GradScaler,
    cfg:        DiffusionConfig,
    epoch:      int,
    logger:     logging.Logger,
    device:     torch.device,
    normalizer: LatentNormalizer | None = None,
    ema_s:      "EMA | None" = None,
    ema_p:      "EMA | None" = None,
) -> dict:
    style_dn.train(); point_dn.train()
    totals    = {}
    n_batches = 0
    gamma = cfg.min_snr_gamma if cfg.min_snr_gamma > 0 else None

    for batch_idx, (z_g, z_local, class_idx) in enumerate(loader):
        # Latents are precomputed (frozen VAE) — just move + normalise.
        z_g       = z_g.to(device, non_blocking=True)
        z_local   = z_local.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)
        if normalizer is not None:
            z_g     = normalizer.norm_zg(z_g)
            z_local = normalizer.norm_zl(z_local)

        # --- Stage 1: style DDPM loss (separate scaler avoids cross-contamination) ---
        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx, gamma)

        opt_s.zero_grad(set_to_none=True)
        scaler_s.scale(loss_s).backward()
        scaler_s.unscale_(opt_s)
        nn.utils.clip_grad_norm_(style_dn.parameters(), cfg.grad_clip)
        scaler_s.step(opt_s)
        scaler_s.update()
        if ema_s is not None:
            ema_s.update(style_dn)

        # --- Stage 2: latent point DDPM loss (conditioned on normalised z_g) ---
        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_p = diffusion_loss(schedule, point_dn, z_local, z_g, gamma)

        opt_p.zero_grad(set_to_none=True)
        scaler_p.scale(loss_p).backward()
        scaler_p.unscale_(opt_p)
        nn.utils.clip_grad_norm_(point_dn.parameters(), cfg.grad_clip)
        scaler_p.step(opt_p)
        scaler_p.update()
        if ema_p is not None:
            ema_p.update(point_dn)

        totals["loss_style"] = totals.get("loss_style", 0.0) + loss_s.item()
        totals["loss_point"] = totals.get("loss_point", 0.0) + loss_p.item()
        n_batches += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            logger.info(
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  "
                + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    schedule:   CosineSchedule,
    style_dn:   StyleDenoiser,
    point_dn:   LatentPointDenoiser,
    loader:     DataLoader,
    cfg:        DiffusionConfig,
    device:     torch.device,
    normalizer: LatentNormalizer | None = None,
) -> dict:
    style_dn.eval(); point_dn.eval()
    totals    = {}
    n_batches = 0
    gamma = cfg.min_snr_gamma if cfg.min_snr_gamma > 0 else None

    for z_g, z_local, class_idx in loader:
        z_g       = z_g.to(device, non_blocking=True)
        z_local   = z_local.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)
        if normalizer is not None:
            z_g     = normalizer.norm_zg(z_g)
            z_local = normalizer.norm_zl(z_local)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx, gamma)
            loss_p = diffusion_loss(schedule, point_dn, z_local, z_g, gamma)

        totals["loss_style"] = totals.get("loss_style", 0.0) + loss_s.item()
        totals["loss_point"] = totals.get("loss_point", 0.0) + loss_p.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ============================================================
# Main
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)


def get_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("diff_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh  = logging.FileHandler(os.path.join(log_dir, "train.log"))
        sh  = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main(cfg: DiffusionConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = torch.device(
        cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"Device: {device}")

    # ---- Load & freeze VAE -----------------------------------------
    if cfg.vae_type == "up":
        vae: Vae | VaeUp = VaeUp(
            latent_dim=cfg.vae_latent_dim,
            style_dim=cfg.vae_style_dim,
            in_channels=cfg.vae_in_channels,
            n_latent=cfg.vae_n_latent,
        ).to(device)
    elif cfg.vae_type == "base":
        vae = Vae(
            latent_dim=cfg.vae_latent_dim,
            style_dim=cfg.vae_style_dim,
            in_channels=cfg.vae_in_channels,
        ).to(device)
    else:
        raise ValueError(f"Unknown vae_type '{cfg.vae_type}'. Choose 'up' or 'base'.")

    if not os.path.exists(cfg.vae_ckpt):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {cfg.vae_ckpt}\n"
            "Run train_vae.py first and pass --vae_ckpt to this script."
        )
    ckpt = torch.load(cfg.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"VAE loaded from {cfg.vae_ckpt}  (frozen)")

    # ---- Data (loaded first: StyleDenoiser needs the real class count) ---
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)

    # class_to_idx is derived from whatever class ids are actually present in
    # cfg.data_root (sorted by id string) — NOT from the full 55-class
    # ShapeNet map(). Building idx_to_name from map() directly, independent
    # of class_to_idx, silently mislabels classes whenever data_root holds a
    # different subset/order (and previously left StyleDenoiser's embedding
    # table oversized with untrained rows if fewer classes were present).
    from src.dataset import Ds_point_model
    name_map        = Ds_point_model.map()   # class id string → readable name
    idx_to_name     = {idx: name_map.get(cid, cid) for cid, idx in base_ds.class_to_idx.items()}
    present_classes = sorted(idx_to_name.keys())

    n_classes_actual = len(base_ds.class_to_idx)
    if n_classes_actual != cfg.num_classes:
        logger.info(
            f"num_classes={cfg.num_classes} in config but {n_classes_actual} class(es) "
            f"found in '{cfg.data_root}' ({[idx_to_name[i] for i in present_classes]}) "
            f"— using {n_classes_actual}."
        )
        cfg.num_classes = n_classes_actual

    # ---- Diffusion models -----------------------------------------
    schedule  = CosineSchedule(T=cfg.T).to(device)

    style_dn  = StyleDenoiser(
        style_dim=cfg.vae_style_dim,
        num_classes=cfg.num_classes,
        hidden=cfg.style_hidden,
        n_layers=cfg.style_layers,
        T=cfg.T,
        cfg_dropout=cfg.cfg_dropout,
    ).to(device)

    point_dn  = LatentPointDenoiser(
        point_dim=cfg.vae_latent_dim,
        style_dim=cfg.vae_style_dim,
        hidden=cfg.point_hidden,
        n_layers=cfg.point_layers,
        n_heads=cfg.point_heads,
        mlp_ratio=cfg.point_mlp_ratio,
        T=cfg.T,
    ).to(device)

    logger.info(f"StyleDenoiser      params: {count_parameters(style_dn):,}")
    logger.info(f"LatentPointDenoiser params: {count_parameters(point_dn):,}")

    indices   = torch.randperm(len(base_ds),
                               generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n     = max(1, int(len(base_ds) * cfg.val_split))
    train_idx = indices[val_n:]          # unchanged vs. original split
    val_pool  = indices[:val_n]          # this pool is split into val + test

    if cfg.test_split > 0:
        # test_split is a fraction of val_pool (not the full dataset),
        # since training indices must not change.
        test_n   = min(
            max(1, int(len(val_pool) * cfg.test_split)),
            len(val_pool) - 1,  # leave at least 1 sample in val
        )
        test_idx = val_pool[:test_n]
        val_idx  = val_pool[test_n:]
    else:
        test_idx = []
        val_idx  = val_pool

    # Point-cloud subsets (augment OFF everywhere — latents are cached once).
    # val_ds/test_ds are still needed as raw points for the MMD/COV/1-NNA caches.
    trn_pts = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), train_idx)
    val_ds  = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)
    test_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), test_idx)
    logger.info(
        f"Dataset: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
    )

    # Precompute latents once (frozen VAE) and train on the cache — removes the
    # VAE forward from every step and makes the target fully deterministic.
    logger.info("Caching VAE latents (one pass, augment off, deterministic FPS)...")
    trn_cache  = build_latent_cache(vae, trn_pts, cfg, device, logger, "train")
    val_cache  = build_latent_cache(vae, val_ds,  cfg, device, logger, "val")
    test_cache = (build_latent_cache(vae, test_ds, cfg, device, logger, "test")
                  if len(test_idx) > 0 else None)

    trn_loader  = DataLoader(trn_cache, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader  = DataLoader(val_cache, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = (DataLoader(test_cache, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
                   if test_cache is not None else None)

    # ---- Reference caches for MMD / COV / 1-NNA -------------------------
    def _build_ref_cache(subset) -> dict:
        cache: dict = {}
        for feats, cls in subset:
            cache.setdefault(int(cls.item()), []).append(feats[:, :3])
        for cls_idx, pts_list in cache.items():
            xyz = torch.stack(pts_list).to(device)
            cache[cls_idx] = normalize_pc(xyz)
        return cache

    logger.info("Building per-class reference caches (val + test)...")
    ref_by_class      = _build_ref_cache(val_ds)   # used during training (eval_every)
    test_ref_by_class = _build_ref_cache(test_ds)  # used only for final test metrics
    logger.info(
        "Val  cache: "
        + ", ".join(f"{idx_to_name.get(k, k)}={v.shape[0]}" for k, v in ref_by_class.items())
    )
    if test_ref_by_class:
        logger.info(
            "Test cache: "
            + ", ".join(f"{idx_to_name.get(k, k)}={v.shape[0]}" for k, v in test_ref_by_class.items())
        )

    # ---- Optimisers (separate per model) ---------------------------
    def make_opt_sched(model):
        opt    = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                          total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs,
                                   eta_min=cfg.lr * 1e-2)
        sched  = SequentialLR(opt, schedulers=[warmup, cosine],
                               milestones=[cfg.warmup_epochs])
        return opt, sched

    opt_s, sched_s = make_opt_sched(style_dn)
    opt_p, sched_p = make_opt_sched(point_dn)
    # Separate scalers prevent style_dn inf/nan from suppressing point_dn updates.
    amp_on   = cfg.amp and device.type == "cuda"
    scaler_s = torch.amp.GradScaler("cuda", enabled=amp_on)
    scaler_p = torch.amp.GradScaler("cuda", enabled=amp_on)

    # ---- EMA dos pesos (amostragem usa a média suavizada) ----
    ema_on = cfg.ema_decay > 0
    ema_s  = EMA(style_dn, cfg.ema_decay) if ema_on else None
    ema_p  = EMA(point_dn, cfg.ema_decay) if ema_on else None
    if ema_on:
        logger.info(f"EMA ativado (decay={cfg.ema_decay}, warmup adaptativo)")

    # ---- Latent normalisation (computed directly from the cached latents) ---
    normalizer: LatentNormalizer | None = None

    # ---- Resume ---------------------------------------------------
    start_epoch    = 0
    best_val_loss  = math.inf
    history: list  = []

    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt_d = torch.load(resume_path, map_location=device)
        style_dn.load_state_dict(ckpt_d["style_dn"])
        point_dn.load_state_dict(ckpt_d["point_dn"])
        opt_s.load_state_dict(ckpt_d["opt_s"])
        opt_p.load_state_dict(ckpt_d["opt_p"])
        sched_s.load_state_dict(ckpt_d["sched_s"])
        sched_p.load_state_dict(ckpt_d["sched_p"])
        scaler_s.load_state_dict(ckpt_d.get("scaler_s", ckpt_d.get("scaler", {})))
        scaler_p.load_state_dict(ckpt_d.get("scaler_p", ckpt_d.get("scaler", {})))
        start_epoch   = ckpt_d["epoch"] + 1
        best_val_loss = ckpt_d.get("best_val_loss", math.inf)
        history       = ckpt_d.get("history", [])
        if "normalizer" in ckpt_d:
            normalizer = LatentNormalizer.from_state_dict(ckpt_d["normalizer"], device)
            logger.info("Normaliser loaded from checkpoint.")
        if ema_on and "ema_s" in ckpt_d:
            ema_s.load_state_dict(ckpt_d["ema_s"], device)
            ema_p.load_state_dict(ckpt_d["ema_p"], device)
            logger.info("EMA loaded from checkpoint.")
        logger.info(f"Resumed from epoch {start_epoch}")

    if normalizer is None:
        normalizer = normalizer_from_cache(trn_cache, device, logger)

    # ---- Training loop --------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn = train_one_epoch(
            schedule, style_dn, point_dn,
            trn_loader, opt_s, opt_p, scaler_s, scaler_p,
            cfg, epoch, logger, device, normalizer, ema_s, ema_p,
        )
        val = validate(schedule, style_dn, point_dn, val_loader, cfg, device, normalizer)

        sched_s.step(); sched_p.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={sched_s.get_last_lr()[0]:.2e}  time={elapsed:.1f}s\n"
            f"  trn: {trn}\n  val: {val}"
        )

        writer.add_scalar("train/lr_style", sched_s.get_last_lr()[0], epoch)
        writer.add_scalar("train/lr_point", sched_p.get_last_lr()[0], epoch)
        log_metrics(writer, trn, "train", epoch)
        log_metrics(writer, val, "val",   epoch)

        # Amostragem/avaliação usam os pesos EMA (restaurados ao sair do bloco).
        ema_pairs = [(ema_s, style_dn), (ema_p, point_dn)] if ema_on else []

        if epoch % cfg.vis_every == 0:
            with use_ema(ema_pairs):
                log_generations(
                    writer, schedule, style_dn, point_dn, vae, cfg, epoch,
                    device, idx_to_name, present_classes, normalizer,
                )

        eval_summary = {}
        if epoch % cfg.eval_every == 0:
            with use_ema(ema_pairs):
                eval_summary = evaluate_generation_metrics(
                    writer, schedule, style_dn, point_dn, vae, cfg, epoch,
                    device, idx_to_name, ref_by_class, logger, normalizer,
                )

        val_loss = val.get("loss_style", math.inf) + val.get("loss_point", math.inf)
        is_best  = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"  New best val loss: {best_val_loss:.5f}")

        save_state = {
            "epoch":         epoch,
            "style_dn":      style_dn.state_dict(),
            "point_dn":      point_dn.state_dict(),
            "opt_s":         opt_s.state_dict(),
            "opt_p":         opt_p.state_dict(),
            "sched_s":       sched_s.state_dict(),
            "sched_p":       sched_p.state_dict(),
            "scaler_s":      scaler_s.state_dict(),
            "scaler_p":      scaler_p.state_dict(),
            "normalizer":    normalizer.state_dict(),
            "ema_s":         ema_s.state_dict() if ema_on else None,
            "ema_p":         ema_p.state_dict() if ema_on else None,
            "best_val_loss": best_val_loss,
            "config":        asdict(cfg),
            "history":       history,
        }
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))
        if is_best:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))

        history.append({"epoch": epoch, **{f"trn_{k}": v for k, v in trn.items()},
                                          **{f"val_{k}": v for k, v in val.items()},
                                          **{f"eval_{k}": v for k, v in eval_summary.items()}})
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ---- Final evaluation on the held-out test set ----------------------
    if test_ref_by_class:
        logger.info("=== Final test-set evaluation ===")
        test_val = validate(
            schedule, style_dn, point_dn, test_loader, cfg, device, normalizer
        )
        log_metrics(writer, test_val, "test", cfg.epochs)
        logger.info(f"  test losses: {test_val}")

        with use_ema([(ema_s, style_dn), (ema_p, point_dn)] if ema_on else []):
            test_gen = evaluate_generation_metrics(
                writer, schedule, style_dn, point_dn, vae, cfg, cfg.epochs,
                device, idx_to_name, test_ref_by_class, logger, normalizer,
            )
        with open(os.path.join(cfg.log_dir, "test_metrics.json"), "w") as f:
            json.dump({"losses": test_val, "generation": test_gen}, f, indent=2)
        logger.info(f"  test metrics saved → {cfg.log_dir}/test_metrics.json")

    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> DiffusionConfig:
    cfg = DiffusionConfig()
    p   = argparse.ArgumentParser(description="Train LION two-stage diffusion")
    for name, val in asdict(cfg).items():
        t = type(val)
        if t is bool:
            p.add_argument(f"--{name}", default=val,
                           type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{name}", default=val, type=t)
    return DiffusionConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)