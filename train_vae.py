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
from src.metric import f_score, normal_consistency, vae_loss
from src.Vae import Vae


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainConfig:
    # --- paths ---
    data_root:     str  = "point_clouds"
    ckpt_dir:      str  = "checkpoints"
    log_dir:       str  = "logs"

    # --- architecture ---
    latent_dim:    int   = 256
    style_dim:     int   = 512
    num_points:    int   = 2048

    # --- training ---
    epochs:        int   = 200
    batch_size:    int   = 32
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0
    normal_weight: float = 1.0

    # --- VAE BETA SCHEDULING ---
    beta_start:    float = 0.0    
    beta_end:      float = 0.001    # Aumentado para equilibrar com o novo cálculo médio da KL
    beta_epochs:   int   = 60     # Estabiliza o espaço latente mais cedo na metade do treino
    recon_loss:    str   = "chamfer"   
    emd_weight:    float = 0.5    

    # --- data ---
    val_split:     float = 0.1
    num_workers:   int   = 4
    pin_memory:    bool  = True

    # --- misc ---
    seed:          int   = 42
    save_every:    int   = 5     # Salva checkpoints a cada 5 épocas para poupar I/O de disco
    log_every:     int   = 50    
    device:        str   = "cuda"
    amp:           bool  = True   
    resume:        int   = 0     


# ============================================================
# TensorBoard helpers
# ============================================================

def log_metrics_tensorboard(writer: SummaryWriter, metrics: dict[str, float], prefix: str, epoch: int):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


@torch.no_grad()
def log_reconstructions(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_items: int = 4,
):
    """Logs Ground Truth vs Reconstruction, and random samples from the Gaussian prior."""
    model.eval()

    # Capture a verification batch
    points, _ = next(iter(loader))
    points = points.to(device)
    target_xyz = points[..., :3]

    # Inference forward pass
    coords_pred, normals_pred, mu, logvar, mu_style, logvar_style = model(points)

    refined = coords_pred.detach().float().cpu()
    target_xyz = target_xyz.detach().float().cpu()

    B = min(max_items, refined.shape[0])

    # Coloring schemes for mesh visualizations
    gt_colors = torch.zeros_like(target_xyz[:B])
    gt_colors[..., 1] = 255  # Green for Ground Truth

    pred_colors = torch.zeros_like(refined[:B])
    pred_colors[..., 0] = 255  # Red for VAE Reconstruction

    writer.add_mesh("reconstruction/ground_truth", vertices=target_xyz[:B], colors=gt_colors, global_step=epoch)
    writer.add_mesh("reconstruction/prediction", vertices=refined[:B], colors=pred_colors, global_step=epoch)

    # --- Generative Prior Sampling ---
    num_pts = getattr(model, "num_latent_points", 1024)
    num_pts = 2048 
    num_gen = 5
    
    generated_xyz, _ = model.generate(num_samples=num_gen, num_points=num_pts, device=device)
    generated_xyz = generated_xyz.detach().float().cpu()

    gen_colors = torch.zeros_like(generated_xyz)
    gen_colors[..., 2] = 255  # Blue for purely synthesized shapes

    writer.add_mesh("samples/random_generation", vertices=generated_xyz, colors=gen_colors, global_step=epoch)
    model.train()


# ============================================================
# Utilities
# ============================================================

def beta_schedule(epoch: int, cfg: TrainConfig) -> float:
    if epoch >= cfg.beta_epochs:
        return cfg.beta_end
    t = epoch / cfg.beta_epochs
    return cfg.beta_start + t * (cfg.beta_end - cfg.beta_start)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vae_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Train / Val steps
# ============================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    cfg:       TrainConfig,
    epoch:     int,
    logger:    logging.Logger,
    device:    torch.device,
) -> dict[str, float]:

    model.train()
    beta = beta_schedule(epoch, cfg)

    totals: dict[str, float] = {}
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        points, _ = data
        points = points.to(device, non_blocking=True)
        target_xyz = points[..., :3]
        normal_target = points[..., 3:]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            coords_pred, normals_pred, mu, logvar, mu_style, logvar_style = model(points)

            losses = vae_loss(
                pred_xyz=coords_pred,
                target_xyz=target_xyz,
                mu_points=mu,
                logvar_points=logvar,
                mu_style=mu_style,
                normal_target=normal_target,
                logvar_style=logvar_style,
                beta=beta,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                normals_pred=normals_pred,
                normal_weight=cfg.normal_weight
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
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  "
                f"β={beta:.3f}  " + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model:  nn.Module,
    loader: DataLoader,
    cfg:    TrainConfig,
    epoch:  int,
    device: torch.device,
) -> dict[str, float]:

    model.eval()
    beta = beta_schedule(epoch, cfg)

    totals: dict[str, float] = {}
    n_batches = 0

    for points, _ in loader:
        points = points.to(device, non_blocking=True)
        target_xyz = points[..., :3]
        target_nrm = points[..., 3:]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            coords_pred, normals_pred, mu_points, logvar_points, mu_style, logvar_style = model(points)

            losses = vae_loss(
                pred_xyz=coords_pred,
                target_xyz=target_xyz,
                mu_points=mu_points,
                logvar_points=logvar_points,
                mu_style=mu_style,
                normal_target=target_nrm,
                logvar_style=logvar_style,
                beta=beta,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                normals_pred=normals_pred,
                normal_weight=cfg.normal_weight
            )

        fs = f_score(coords_pred, target_xyz, threshold=0.01)
        nc = normal_consistency(coords_pred, normals_pred, target_xyz, target_nrm)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["f_score"] = totals.get("f_score", 0.0) + fs["f_score"].item()
        totals["normal_c"] = totals.get("normal_c", 0.0) + nc.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ============================================================
# Main Execution
# ============================================================

def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Datasets & Splits ---------------------------------------------
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices = torch.randperm(len(base_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n = max(1, int(len(base_ds) * cfg.val_split))

    train_idx = indices[val_n:]
    val_idx   = indices[:val_n]

    trn_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=True), train_idx)
    val_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)
   
    # FIX: shuffle=True ativado no Loader de treinamento para evitar overfitting sequencial
    trn_loader = DataLoader(
        trn_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")

    # ---- Model Instantiation -------------------------------------------
    model = Vae(
        latent_dim=cfg.latent_dim, 
        style_dim=cfg.style_dim, 
        in_channels=6
    ).to(device)
    
    logger.info("Hierarchical VAE Initialised successfully.")
    logger.info(f"Parameters: {count_parameters(model):,}")

    # ---- Optimiser & Schedulers ----------------------------------------
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimiser, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume Checkpoint State ---------------------------------------
    start_epoch = 0
    best_val_cd = math.inf
    history: list[dict] = []

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

    # ---- Core Training Epoch Loop --------------------------------------
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
            f"β={beta_schedule(epoch, cfg):.3f}  "
            f"time={elapsed:.1f}s\n"
            f"  {trn_str}\n"
            f"  {val_str}"
        )

        # TensorBoard logging
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/beta", beta_schedule(epoch, cfg), epoch)

        log_metrics_tensorboard(writer, trn_metrics, "train", epoch)
        log_metrics_tensorboard(writer, val_metrics, "val", epoch)
        writer.add_scalar("latent/best_val_cd", best_val_cd, epoch)

        # Render geometric updates inside TensorBoard
        if epoch % 5 == 0:
            log_reconstructions(writer, model, trn_loader, device, epoch, max_items=4)

        # ---- Track Performance & Save Checkpoints ----------------------
        record = {
            "epoch": epoch, 
            **{f"trn_{k}": v for k, v in trn_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(record)

        # Reads exclusively the "recon" key mapping Chamfer/EMD distance
        val_cd = val_metrics.get("recon", math.inf)
        is_best = val_cd < best_val_cd
        if is_best:
            best_val_cd = val_cd
            logger.info(f"  ✓ New best val reconstruction (Chamfer): {best_val_cd:.6f}")

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

        # Backup state to prevent tracking losses if training crashes
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))

        if is_best:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))

        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI Parser
# ============================================================

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(description="Train LION Hierarchical Point-Cloud VAE")
    for field_name, field_val in asdict(cfg).items():
        t = type(field_val)
        if t is bool:
            p.add_argument(f"--{field_name}", default=field_val, type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{field_name}", default=field_val, type=t)
    
    args = vars(p.parse_args())
    return TrainConfig(**args)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)