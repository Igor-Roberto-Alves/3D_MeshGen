from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import math
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

from src.dataset import Ds_point_sampled_already
from src.Vae import Vae
from src.Diffus import (
    DDPMScheduler, StyleScoreNet, PointScoreNet, Model
)


@dataclass
class DiffusionConfig:
    # --- paths ---
    data_root:      str   = "point_clouds"
    vae_ckpt:       str   = "checkpoints/best.pt"   # frozen VAE checkpoint
    ckpt_dir:       str   = "checkpoints_diffusion"
    log_dir:        str   = "logs_diffusion"

    # --- VAE architecture (must match the frozen checkpoint) ---
    latent_dim:     int   = 6
    style_dim:      int   = 512

    # --- diffusion architecture ---
    hidden_dim:     int   = 512
    style_depth:    int   = 6
    point_depth:    int   = 8
    heads:          int   = 4
    num_classes:    int   = 53
    T:              int   = 1000
    schedule:       str   = "cosine"   # "cosine" | "linear"

    # --- loss weights ---
    w_style:        float = 1.0
    w_points:       float = 1.0

    # --- training ---
    epochs:         int   = 400
    batch_size:     int   = 32
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0

    # --- data ---
    val_split:      float = 0.1
    num_workers:    int   = 4
    pin_memory:     bool  = True

    # --- misc ---
    seed:           int   = 495241
    save_every:     int   = 2
    log_every:      int   = 50
    device:         str   = "cuda"
    amp:            bool  = True
    resume:         int   = 0


# ------------------------------------------------------------------ #
# Helpers (same pattern as your VAE train.py)
# ------------------------------------------------------------------ #

def log_metrics_tensorboard(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("diffusion_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ #
# Dataset — returns (points, cls) tuples
# ------------------------------------------------------------------ #

def build_loaders(cfg: DiffusionConfig):
    """
    Ds_point_sampled_already must return (points, cls) or you wrap it below.
    If your dataset only returns points, see the wrapper comment.
    """
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices = torch.randperm(
        len(base_ds), generator=torch.Generator().manual_seed(cfg.seed)
    ).tolist()

    val_n     = max(1, int(len(base_ds) * cfg.val_split))
    train_idx = indices[val_n:]
    val_idx   = indices[:val_n]

    trn_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=True), train_idx
    )
    val_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx
    )

    trn_loader = DataLoader(
        trn_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    return trn_loader, val_loader, len(train_idx), len(val_idx)


# ------------------------------------------------------------------ #
# Train / Val steps
# ------------------------------------------------------------------ #

def train_one_epoch(
    model, loader, optimiser, scaler, cfg, epoch, logger, device
) -> dict:
    model.train()
    # VAE must stay frozen/eval even during model.train()
    model.vae.eval()

    totals: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        # --- unpack: adapt if your dataset returns only points ---
        if isinstance(batch, (list, tuple)):
            points, cls = batch
        else:
            points = batch
            cls = None   # should not happen if dataset has classes

        points = points.to(device, non_blocking=True)
        cls    = cls.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            losses = model.training_step(
                points, cls,
                w_style=cfg.w_style,
                w_points=cfg.w_points,
            )

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(
            list(model.style_net.parameters()) +
            list(model.point_net.parameters()),
            cfg.grad_clip,
        )
        scaler.step(optimiser)
        scaler.update()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            logger.info(
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  "
                + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, cfg, epoch, device) -> dict:
    model.eval()

    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            points, cls = batch
        else:
            points = batch
            cls = None

        points = points.to(device, non_blocking=True)
        cls    = cls.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            losses = model.training_step(
                points, cls,
                w_style=cfg.w_style,
                w_points=cfg.w_points,
            )

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def log_samples(writer, model, cfg, device, epoch, n=4):
    """Generate n samples per class and log to TensorBoard."""
    model.eval()

    cls = torch.arange(n, device=device) % cfg.num_classes
    gen_xyz = model.sample(
        n=n, cls=cls, device=device,
        style_dim=cfg.style_dim,
        latent_dim=cfg.latent_dim * cfg.latent_dim,  # adapt to your z_points dim
    )
    gen_xyz = gen_xyz[..., :3].detach().float().cpu()

    colors = torch.zeros_like(gen_xyz)
    colors[..., 2] = 255   # blue

    writer.add_mesh("samples/generated", vertices=gen_xyz, colors=colors, global_step=epoch)
    model.train()
    model.vae.eval()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(cfg: DiffusionConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = torch.device(
        cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ---- dataset -------------------------------------------------------
    trn_loader, val_loader, n_trn, n_val = build_loaders(cfg)
    logger.info(f"Dataset: {n_trn} train / {n_val} val samples")

    # ---- frozen VAE ----------------------------------------------------
    vae = Vae(latent_dim=cfg.latent_dim, style_dim=cfg.style_dim, in_channels=6).to(device)
    ckpt = torch.load(cfg.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt["model"])
    logger.info(f"Loaded VAE from {cfg.vae_ckpt} (epoch {ckpt.get('epoch', '?')})")

    # ---- diffusion models ----------------------------------------------
    style_net = StyleScoreNet(
        style_dim=cfg.style_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.style_depth,
        heads=cfg.heads,
        num_classes=cfg.num_classes,
    ).to(device)

    point_net = PointScoreNet(
        latent_dim=3 + cfg.latent_dim,
        style_dim=cfg.style_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.point_depth,
        heads=cfg.heads,
    ).to(device)

    style_sched = DDPMScheduler(T=cfg.T, schedule=cfg.schedule)
    point_sched = DDPMScheduler(T=cfg.T, schedule=cfg.schedule)

    model = Model(
        frozen_vae=vae,
        style_score_net=style_net,
        point_score_net=point_net,
        style_scheduler=style_sched,
        point_scheduler=point_sched,
    ).to(device)

    # only style_net + point_net are trainable
    trainable = list(model.style_net.parameters()) + list(model.point_net.parameters())
    logger.info("*******************************************")
    logger.info("Diffusion Model initialised.")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    logger.info("*******************************************")

    # ---- optimiser / scheduler -----------------------------------------
    optimiser = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimiser, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- resume --------------------------------------------------------
    start_epoch = 0
    best_val    = math.inf
    history     = []

    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt = torch.load(resume_path, map_location=device)
        model.style_net.load_state_dict(ckpt["style_net"])
        model.point_net.load_state_dict(ckpt["point_net"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", math.inf)
        history     = ckpt.get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- training loop -------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn_metrics = train_one_epoch(model, trn_loader, optimiser, scaler, cfg, epoch, logger, device)
        val_metrics = validate(model, val_loader, cfg, epoch, device)

        scheduler.step()
        elapsed = time.time() - t0

        trn_str = "  ".join(f"trn_{k}={v:.5f}" for k, v in trn_metrics.items())
        val_str = "  ".join(f"val_{k}={v:.5f}" for k, v in val_metrics.items())
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.1f}s\n  {trn_str}\n  {val_str}"
        )

        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        log_metrics_tensorboard(writer, trn_metrics, "train", epoch)
        log_metrics_tensorboard(writer, val_metrics, "val", epoch)

        if epoch % 5 == 0:
            log_samples(writer, model, cfg, device, epoch)

        # ---- checkpoint ------------------------------------------------
        record = {"epoch": epoch,
                  **{f"trn_{k}": v for k, v in trn_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        save_state = {
            "epoch":      epoch,
            "style_net":  model.style_net.state_dict(),
            "point_net":  model.point_net.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_val":   best_val,
            "config":     asdict(cfg),
            "history":    history,
        }
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))

        val_loss = val_metrics.get("loss", math.inf)
        if val_loss < best_val:
            best_val = val_loss
            save_state["best_val"] = best_val
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))
            logger.info(f"  ✓ New best val loss: {best_val:.6f}")

        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    logger.info("Training complete.")


def parse_args() -> DiffusionConfig:
    cfg = DiffusionConfig()
    p   = argparse.ArgumentParser(description="Train Diffusion prior over VAE latents")
    for field_name, field_val in asdict(cfg).items():
        t = type(field_val)
        if t is bool:
            p.add_argument(f"--{field_name}", default=field_val, type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{field_name}", default=field_val, type=t)
    return DiffusionConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)