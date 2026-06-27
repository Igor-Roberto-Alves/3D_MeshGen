"""
train_vae.py
------------
Training loop for the LION-inspired Hierarchical Point-Cloud VAE.
"""

import argparse
import json
import logging
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
from src.metric import f_score, vae_loss
from src.Vae import Vae, normalize_pc


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainConfig:
    # --- paths ---
    data_root:     str   = "point_clouds"
    ckpt_dir:      str   = "checkpoints"
    log_dir:       str   = "logs"

    # --- architecture ---
    latent_dim:    int   = 3      # per-point latent features (LION paper: ~3)
    style_dim:     int   = 256    # global shape latent dim
    in_channels:   int   = 6

    # --- training ---
    epochs:        int   = 200
    batch_size:    int   = 16
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0

    # --- VAE beta scheduling (KL annealing) ---
    beta_start:    float = 0.0
    beta_end:      float = 1.0
    beta_epochs:   int   = 150
    # Per-latent KL weights — mirrors LION's weight_kl_pt / weight_kl_feat / weight_kl_glb.
    # beta_style:    z_g KL weight  — 1.0: z_g is a standard VAE latent; DDPM level 1
    #                                  generates it from N(0,I).
    # beta_pos_pts:  z_l position channel KL (first 3 dims).  Keep near 0; the
    #                position anchor provides the reconstruction signal — KL here
    #                fights the anchor and hurts CD.
    # beta_feat_pts: z_l feature channel KL (remaining latent_dim dims).  Mild
    #                regularisation (0.001) helps the diffusion model without
    #                hurting reconstruction; matches LION's weight_kl_feat.
    beta_style:     float = 1.0
    beta_pos_pts:   float = 1e-4
    beta_feat_pts:  float = 0.001

    # Position noise injected into z_l[:,:,:3] before decoding.
    # Breaks the position shortcut so the decoder must use z_g for correction.
    # 0.0 = disabled; 0.05 is a good starting value (≈5% of normalised range).
    pos_noise_std:  float = 0.05

    # --- reconstruction loss ---
    recon_loss:    str   = "chamfer"
    emd_weight:    float = 0.5

    # --- data ---
    val_split:     float = 0.1
    num_workers:   int   = 4
    pin_memory:    bool  = True

    # --- normal loss ---
    normal_weight: float = 0.1   # weight for predicted-vs-GT normal consistency loss

    # --- misc ---
    seed:          int   = 42
    save_every:    int   = 5
    log_every:     int   = 50
    device:        str   = "cuda"
    amp:           bool  = True
    resume:        int   = 0


# ============================================================
# TensorBoard helpers
# ============================================================

def log_metrics_tensorboard(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


def _normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) unit normals → (B, N, 3) uint8 RGB using abs values."""
    return (normals.abs() * 255).clamp(0, 255).to(torch.uint8)


# ---- GT / Reconstruction visualisation (separate pages for train & val) ----

@torch.no_grad()
def log_gt_recon(writer, model, loader, device, epoch, tag_prefix: str, max_items: int = 4):
    """
    Log GT and reconstruction as separate meshes (no side-by-side crowding).
    Tags:
        recon/{tag_prefix}/gt/sample_{i}    — green  — ground truth
        recon/{tag_prefix}/recon/sample_{i} — red    — reconstruction
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)
    N = points.shape[1]

    # Use posterior means (no reparameterization noise) for a clean visual.
    # Training forward() always samples z=mu+eps for the ELBO; at visualization
    # time that noise (sigma~0.7-0.8) corrupts xyz_init and ruins the picture
    # even when mu_l is a good encoding of the shape.
    xyz_out, _ = model.reconstruct(points)
    gt    = normalize_pc(points)[..., :3].detach().float().cpu()
    recon = xyz_out.detach().float().cpu()
    B     = min(max_items, gt.shape[0])


    green = torch.tensor([[0, 220, 0]],   dtype=torch.uint8).expand(N, -1)
    red   = torch.tensor([[220, 0, 0]],   dtype=torch.uint8).expand(N, -1)

    for i in range(B):
        writer.add_mesh(
            f"recon/{tag_prefix}/gt/sample_{i}",
            vertices=gt[i].unsqueeze(0),
            colors=green.unsqueeze(0),
            global_step=epoch,
        )
        writer.add_mesh(
            f"recon/{tag_prefix}/recon/sample_{i}",
            vertices=recon[i].unsqueeze(0),
            colors=red.unsqueeze(0),
            global_step=epoch,
        )
    model.train()


# ---- Style samples (fixed z_l, varying z_g) --------------------------------

@torch.no_grad()
def log_style_samples(
    writer, model, device, epoch, num_points: int,
    n_fixed_latents: int = 3, n_styles: int = 5, latent_seed: int = 0,
):
    """
    Fix n_fixed_latents z_l vectors (deterministic across epochs via latent_seed),
    then decode each with n_styles fresh z_g samples drawn from the prior.

    This shows how the global style vector controls shape while the local
    per-point structure is held constant.

    Tags: samples/latent_{i}/style_{j}
    """
    model.eval()

    # Fixed z_l — constant across all epochs (shape: B, N, 3+latent_dim)
    gen = torch.Generator()
    gen.manual_seed(latent_seed)
    fixed_zl = torch.randn(
        n_fixed_latents, num_points, model.total_z_dim, generator=gen,
    ).clamp(-1, 1).to(device)

    blue = torch.tensor([[60, 120, 220]], dtype=torch.uint8)

    for i in range(n_fixed_latents):
        zl = fixed_zl[i].unsqueeze(0)                                        # (1, N, latent_dim)
        for j in range(n_styles):
            zg       = torch.randn(1, model.style_dim, device=device)
            pts, _   = model.decoder(zl, zg)
            pts      = pts.detach().float().cpu().squeeze(0)                  # (N, 3)
            clr = blue.expand(pts.shape[0], -1).unsqueeze(0)                 # (1, N, 3)
            writer.add_mesh(
                f"samples/latent_{i}/style_{j}",
                vertices=pts.unsqueeze(0),
                colors=clr,
                global_step=epoch,
            )
    model.train()


# ---- Normal map comparison -------------------------------------------------

@torch.no_grad()
def log_normal_maps(writer, model, loader, device, epoch, tag_prefix: str, max_items: int = 4):
    """
    Visualise surface normals of GT and reconstructed clouds side-by-side.

    GT normals   : taken directly from the input data (channels 3-5), unit-normalised.
    Recon normals: estimated from the reconstructed xyz via PCA on k-NN (see
                   estimate_normals_pca).  The decoder does not output normals; PCA is
                   the correct way to derive them from the predicted point positions.

    Points are coloured by abs(normal) → RGB so that sign-inconsistency in PCA
    normals does not distort the comparison.

    Tags:
        normals/{tag_prefix}/gt/sample_{i}    — ground truth normal colours
        normals/{tag_prefix}/recon/sample_{i} — reconstructed normal colours
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)

    # Use posterior means — same reason as log_gt_recon.
    xyz_out, normals_out = model.reconstruct(points)
    pts_norm  = normalize_pc(points).detach().float().cpu()

    gt_xyz    = pts_norm[..., :3]
    gt_nrm    = F.normalize(pts_norm[..., 3:6], dim=-1)
    recon_xyz = xyz_out.detach().float().cpu()
    recon_nrm = normals_out.detach().float().cpu()    # decoder predicts normals directly

    B = min(max_items, gt_xyz.shape[0])
    for i in range(B):
        gt_rgb    = _normals_to_rgb(gt_nrm[i].unsqueeze(0))     # (1, N, 3)
        recon_rgb = _normals_to_rgb(recon_nrm[i].unsqueeze(0))

        writer.add_mesh(
            f"normals/{tag_prefix}/gt/sample_{i}",
            vertices=gt_xyz[i].unsqueeze(0),
            colors=gt_rgb,
            global_step=epoch,
        )
        writer.add_mesh(
            f"normals/{tag_prefix}/recon/sample_{i}",
            vertices=recon_xyz[i].unsqueeze(0),
            colors=recon_rgb,
            global_step=epoch,
        )
    model.train()


# ---- Latent space analysis (active units) ----------------------------------

@torch.no_grad()
def log_latent_analysis(writer, model, loader, device, epoch):
    """
    Run one batch through the encoder and log per-dimension KL for z_l and z_g.

    Active units: dimensions where mean KL > 0.1 — these are the ones the
    encoder is actually using to encode information.

    If active_units_local == latent_dim  → all dims are used, size is fine.
    If active_units_local << latent_dim  → latent_dim is too large (waste).
    If active_units_local == latent_dim AND recon is still high → too small.
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)

    _, _, mu_l, logvar_l, mu_g, logvar_g = model(points)

    # Per-dimension KL  (mean over batch and, for z_l, over N points)
    # kl_elem shape: same as mu
    def kl_per_dim(mu, logvar):
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())   # (..., D)
        # reduce everything except the last dim
        while kl.dim() > 1:
            kl = kl.mean(dim=0)
        return kl   # (D,)

    kl_local  = kl_per_dim(mu_l,  logvar_l)   # (latent_dim,)
    kl_global = kl_per_dim(mu_g,  logvar_g)   # (style_dim,)

    threshold = 0.1
    active_local  = (kl_local  > threshold).sum().item()
    active_global = (kl_global > threshold).sum().item()

    writer.add_scalar("latent/active_units_local",  active_local,  epoch)
    writer.add_scalar("latent/active_units_global", active_global, epoch)
    writer.add_scalar("latent/latent_dim",  mu_l.shape[-1],  epoch)
    writer.add_scalar("latent/style_dim",   mu_g.shape[-1],  epoch)

    # Per-dimension KL as individual scalars (readable in TensorBoard)
    for d, v in enumerate(kl_local.tolist()):
        writer.add_scalar(f"latent/kl_local_dim{d}", v, epoch)
    # For style_dim only log first 32 dims to keep TensorBoard tidy
    for d, v in enumerate(kl_global[:32].tolist()):
        writer.add_scalar(f"latent/kl_global_dim{d}", v, epoch)

    # Histogram of per-dim KL (most compact summary)
    writer.add_histogram("latent/kl_local_hist",  kl_local,  epoch)
    writer.add_histogram("latent/kl_global_hist", kl_global, epoch)

    model.train()


# ---- Gradient norm logging -------------------------------------------------

def log_gradient_norms(writer, model, global_step: int):
    """
    Log per-module gradient L2 norms and the overall pre-clip norm.
    Call after scaler.unscale_() so gradients are in their true fp32 scale.
    """
    modules = {
        "global_encoder": model.global_encoder,
        "local_encoder":  model.local_encoder,
        "decoder":        model.decoder,
    }
    total_sq = 0.0
    for name, mod in modules.items():
        sq = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in mod.parameters()
            if p.grad is not None
        )
        writer.add_scalar(f"gradients/{name}", sq ** 0.5, global_step)
        total_sq += sq
    writer.add_scalar("gradients/total_before_clip", total_sq ** 0.5, global_step)


# ============================================================
# Utilities
# ============================================================

def beta_schedule(epoch, cfg):
    if epoch >= cfg.beta_epochs:
        return cfg.beta_end
    t = epoch / cfg.beta_epochs
    return cfg.beta_start + t * (cfg.beta_end - cfg.beta_start)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vae_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh  = logging.FileHandler(os.path.join(log_dir, "train.log"))
        sh  = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Train / Val steps
# ============================================================

def train_one_epoch(model, loader, optimiser, scaler, cfg, epoch, logger, device, writer=None):
    model.train()
    beta      = beta_schedule(epoch, cfg)
    totals    = {}
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        points, _ = data
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, normals_out, mu_l, logvar_l, mu_g, logvar_g = model(
                points, pos_noise_std=cfg.pos_noise_std)
            target_nrm = F.normalize(normalize_pc(points)[..., 3:6], dim=-1)
            losses = vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu_points=mu_l,
                logvar_points=logvar_l,
                mu_style=mu_g,
                logvar_style=logvar_g,
                beta=beta,
                beta_pos_pts=cfg.beta_pos_pts,
                beta_feat_pts=cfg.beta_feat_pts,
                beta_style=cfg.beta_style,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                normals_pred=normals_out,
                normal_target=target_nrm,
                normal_weight=cfg.normal_weight,
            )

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimiser)

        # Gradient norms logged after unscaling, before clipping
        if writer is not None and (batch_idx + 1) % cfg.log_every == 0:
            log_gradient_norms(writer, model, epoch * len(loader) + batch_idx)

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimiser)
        scaler.update()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            logger.info(
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  β={beta:.4f}  "
                + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, cfg, epoch, device):
    model.eval()
    beta      = beta_schedule(epoch, cfg)
    totals    = {}
    n_batches = 0

    for points, _ in loader:
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, normals_out, mu_l, logvar_l, mu_g, logvar_g = model(
                points, pos_noise_std=cfg.pos_noise_std)
            target_nrm = F.normalize(normalize_pc(points)[..., 3:6], dim=-1)
            losses = vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu_points=mu_l,
                logvar_points=logvar_l,
                mu_style=mu_g,
                logvar_style=logvar_g,
                beta=beta,
                beta_pos_pts=cfg.beta_pos_pts,
                beta_feat_pts=cfg.beta_feat_pts,
                beta_style=cfg.beta_style,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                normals_pred=normals_out,
                normal_target=target_nrm,
                normal_weight=cfg.normal_weight,
            )

        fs = f_score(xyz_out, target_xyz, threshold=0.01)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["f_score"] = totals.get("f_score", 0.0) + fs["f_score"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ============================================================
# Main
# ============================================================

def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Datasets -------------------------------------------------------
    base_ds  = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices  = torch.randperm(len(base_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n    = max(1, int(len(base_ds) * cfg.val_split))
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    trn_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=True),  train_idx)
    val_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)

    trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")

    # num_points from the dataset (needed for style-sample generation)
    sample_pts, _ = trn_ds[0]
    num_points = sample_pts.shape[0]

    # ---- Model ----------------------------------------------------------
    model = Vae(
        latent_dim=cfg.latent_dim,
        style_dim=cfg.style_dim,
        in_channels=cfg.in_channels,
    ).to(device)
    logger.info(f"LION VAE  |  params: {count_parameters(model):,}")
    logger.info(f"  latent_dim={cfg.latent_dim}  style_dim={cfg.style_dim}")

    # ---- Optimiser & scheduler ------------------------------------------
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup    = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine    = CosineAnnealingLR(optimiser, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume ---------------------------------------------------------
    start_epoch  = 0
    best_val_cd  = math.inf
    history: list = []

    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_cd = ckpt.get("best_val_cd", math.inf)
        history     = ckpt.get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop --------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0          = time.time()
        trn_metrics = train_one_epoch(
            model, trn_loader, optimiser, scaler, cfg, epoch, logger, device, writer=writer,
        )
        val_metrics = validate(model, val_loader, cfg, epoch, device)
        scheduler.step()

        elapsed = time.time() - t0
        trn_str = "  ".join(f"trn_{k}={v:.5f}" for k, v in trn_metrics.items())
        val_str = "  ".join(f"val_{k}={v:.5f}" for k, v in val_metrics.items())
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  β={beta_schedule(epoch, cfg):.4f}  "
            f"time={elapsed:.1f}s\n  {trn_str}\n  {val_str}"
        )

        writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/beta", beta_schedule(epoch, cfg),  epoch)
        log_metrics_tensorboard(writer, trn_metrics, "train", epoch)
        log_metrics_tensorboard(writer, val_metrics, "val",   epoch)

        if epoch % 5 == 0:
            log_latent_analysis(writer, model, val_loader, device, epoch)

            # Ground truth vs reconstruction — train and val separately
            log_gt_recon(writer, model, trn_loader, device, epoch, "train", max_items=4)
            log_gt_recon(writer, model, val_loader,  device, epoch, "val",  max_items=4)

            # Normal map comparison — train and val separately
            log_normal_maps(writer, model, trn_loader, device, epoch, "train", max_items=4)
            log_normal_maps(writer, model, val_loader,  device, epoch, "val",  max_items=4)

            # Style samples: fixed z_l, varying z_g
            log_style_samples(
                writer, model, device, epoch,
                num_points=num_points, n_fixed_latents=3, n_styles=5, latent_seed=0,
            )

        val_cd = val_metrics.get("recon", math.inf)
        is_best = val_cd < best_val_cd
        if is_best:
            best_val_cd = val_cd
            logger.info(f"  New best val recon: {best_val_cd:.6f}")

        save_state = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimiser":   optimiser.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict(),
            "best_val_cd": best_val_cd,
            "config":      asdict(cfg),
            "history":     history,
        }
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))
        if is_best:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))

        history.append({"epoch": epoch,
                        **{f"trn_{k}": v for k, v in trn_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(description="Train LION Hierarchical Point-Cloud VAE")
    for field_name, field_val in asdict(cfg).items():
        t = type(field_val)
        if t is bool:
            p.add_argument(f"--{field_name}", default=field_val,
                           type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{field_name}", default=field_val, type=t)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
