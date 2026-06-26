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
    beta_end:      float = 2.0
    beta_epochs:   int   = 80

    # --- reconstruction loss ---
    recon_loss:    str   = "chamfer"
    emd_weight:    float = 0.5

    # --- data ---
    val_split:     float = 0.1
    num_workers:   int   = 4
    pin_memory:    bool  = True

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


def _side_by_side(clouds: list[torch.Tensor], colors: list[torch.Tensor], gap: float = 2.5):
    """
    Concatenate N point clouds into one mesh, offset along X so they sit side-by-side.
    clouds: list of (N_pts, 3) tensors (already CPU float)
    colors: list of matching (N_pts, 3) uint8 tensors
    Returns vertices (1, N_total, 3) and colors (1, N_total, 3) ready for add_mesh.
    """
    shifted_v, shifted_c = [], []
    x_cursor = 0.0
    for v, c in zip(clouds, colors):
        v = v.clone()
        # centre each cloud, then shift by cursor
        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += x_cursor
        half_width = (v[:, 0].max() - v[:, 0].min()).item() * 0.5
        x_cursor  += half_width * 2 + gap
        shifted_v.append(v)
        shifted_c.append(c)
    verts  = torch.cat(shifted_v, dim=0).unsqueeze(0)   # (1, N_total, 3)
    clrs   = torch.cat(shifted_c, dim=0).unsqueeze(0)   # (1, N_total, 3)
    return verts, clrs


@torch.no_grad()
def log_reconstructions(writer, model, loader, device, epoch, max_items=4):
    """
    For each of max_items samples: log a single mesh showing
        [GT (green) | Reconstruction (red) | Prior sample (blue)]
    side-by-side so all three are always synchronised in the same panel.
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)
    N = points.shape[1]

    xyz_out, *_ = model(points)
    gt    = normalize_pc(points)[..., :3].detach().float().cpu()
    recon = xyz_out.detach().float().cpu()
    B     = min(max_items, gt.shape[0])

    # One prior sample per displayed shape (same N)
    prior_samples = model.generate(num_samples=B, num_points=N, device=device)
    prior_samples = prior_samples.detach().float().cpu()

    for i in range(B):
        gt_v   = gt[i];    gt_c   = torch.tensor([[0, 220, 0]],   dtype=torch.uint8).expand(N, -1)
        rec_v  = recon[i]; rec_c  = torch.tensor([[220, 0, 0]],   dtype=torch.uint8).expand(N, -1)
        pri_v  = prior_samples[i]
        pri_c  = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(N, -1)

        verts, clrs = _side_by_side([gt_v, rec_v, pri_v], [gt_c, rec_c, pri_c])
        writer.add_mesh(f"sample_{i}/gt_recon_prior", vertices=verts, colors=clrs, global_step=epoch)

    model.train()


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

def train_one_epoch(model, loader, optimiser, scaler, cfg, epoch, logger, device):
    model.train()
    beta    = beta_schedule(epoch, cfg)
    totals  = {}
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        points, _ = data
        points     = points.to(device, non_blocking=True)
        # Normalize before extracting target so it matches the model's output space.
        # model(points) calls normalize_pc internally; the loss must compare in the same scale.
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, mu_l, logvar_l, mu_g, logvar_g = model(points)
            losses = vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu_points=mu_l,
                logvar_points=logvar_l,
                mu_style=mu_g,
                logvar_style=logvar_g,
                beta=beta,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
            )

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimiser)
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
    beta    = beta_schedule(epoch, cfg)
    totals  = {}
    n_batches = 0

    for points, _ in loader:
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, mu_l, logvar_l, mu_g, logvar_g = model(points)
            losses = vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu_points=mu_l,
                logvar_points=logvar_l,
                mu_style=mu_g,
                logvar_style=logvar_g,
                beta=beta,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
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
        ckpt = torch.load(resume_path, map_location=device)
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
        trn_metrics = train_one_epoch(model, trn_loader, optimiser, scaler, cfg, epoch, logger, device)
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
            log_reconstructions(writer, model, val_loader, device, epoch, max_items=4)

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
