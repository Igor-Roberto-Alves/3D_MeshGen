"""
train_vae.py
------------
Training loop for the flat two-vector Point-Cloud VAE (Real_latent branch).

  z_g ∈ ℝ^{style_dim}   — global style (GlobalEncoder)
  z_l ∈ ℝ^{latent_size} — flat shape code (ShapeEncoder, no coordinate shortcut)

Both KL terms use the same annealing schedule and per-component beta weights.
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
from src.metric import chamfer_distance_knn, f_score, kl_divergence
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
    latent_size:   int   = 1024    # flat shape latent dimension
    style_dim:     int   = 128     # global style vector
    in_channels:   int   = 6
    num_points:    int   = 2048

    # --- training ---
    epochs:        int   = 200
    batch_size:    int   = 16
    lr:            float = 1e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0

    # --- KL annealing ---
    beta_start:    float = 1e-7
    beta_end:      float = 0.1
    beta_epochs:   int   = 150
    beta_style:    float = 1.0   # weight on KL(z_g)
    beta_shape:    float = 1.0   # weight on KL(z_l)

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
    amp:           bool  = False
    resume:        int   = 0


# ============================================================
# TensorBoard helpers
# ============================================================

def log_scalars(writer, metrics: dict, group: str, epoch: int):
    for k, v in metrics.items():
        writer.add_scalar(f"losses/{group}/{k}", v, epoch)


@torch.no_grad()
def log_gt_recon(writer, model, loader, device, epoch, tag: str, max_items: int = 4):
    """Green = GT,  Red = reconstruction (posterior means, no noise)."""
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)
    N = points.shape[1]

    xyz_out = model.reconstruct(points)
    gt    = normalize_pc(points)[..., :3].detach().float().cpu()
    recon = xyz_out.detach().float().cpu()
    B     = min(max_items, gt.shape[0])

    green = torch.tensor([[0, 200, 0]],   dtype=torch.uint8).expand(N, -1)
    red   = torch.tensor([[200, 0, 0]],   dtype=torch.uint8).expand(N, -1)

    for i in range(B):
        writer.add_mesh(f"recon/{tag}/gt/sample_{i}",
                        vertices=gt[i].unsqueeze(0),
                        colors=green.unsqueeze(0), global_step=epoch)
        writer.add_mesh(f"recon/{tag}/recon/sample_{i}",
                        vertices=recon[i].unsqueeze(0),
                        colors=red.unsqueeze(0), global_step=epoch)
    model.train()


@torch.no_grad()
def log_zg_ablation(writer, model, loader, device, epoch, max_items: int = 4):
    """
    Diagnostic: does z_g actually influence the output?

    Same z_l, only z_g changes:
      GREEN  = GT
      BLUE   = decode with real z_g (posterior mean)
      ORANGE = decode with random z_g ~ N(0,I)

    cd_ratio = cd_random / cd_real:  1.0 = z_g useless,  >1 = z_g helps.
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points[:max_items].to(device)

    x_norm     = normalize_pc(points)
    target_xyz = x_norm[..., :3]

    mu_g, _ = model.style_encoder(x_norm)
    mu_l, _ = model.shape_encoder(x_norm)

    xyz_real = model.decoder(mu_l, mu_g)

    z_g_rand = torch.randn_like(mu_g)
    xyz_rand = model.decoder(mu_l, z_g_rand)

    cd_real = chamfer_distance_knn(xyz_real, target_xyz, reduce="mean")[0].item()
    cd_rand = chamfer_distance_knn(xyz_rand, target_xyz, reduce="mean")[0].item()
    ratio   = cd_rand / (cd_real + 1e-8)

    writer.add_scalar("zg_ablation/cd_real_zg",   cd_real, epoch)
    writer.add_scalar("zg_ablation/cd_random_zg", cd_rand, epoch)
    writer.add_scalar("zg_ablation/cd_ratio",     ratio,   epoch)

    N      = points.shape[1]
    green  = torch.tensor([[0, 200, 0]],   dtype=torch.uint8).expand(N, -1)
    blue   = torch.tensor([[50, 100, 220]], dtype=torch.uint8).expand(N, -1)
    orange = torch.tensor([[220, 130, 0]], dtype=torch.uint8).expand(N, -1)

    gt_cpu   = target_xyz.detach().float().cpu()
    real_cpu = xyz_real.detach().float().cpu()
    rand_cpu = xyz_rand.detach().float().cpu()

    for i in range(gt_cpu.shape[0]):
        writer.add_mesh(f"zg_ablation/gt/sample_{i}",
                        vertices=gt_cpu[i].unsqueeze(0),
                        colors=green.unsqueeze(0), global_step=epoch)
        writer.add_mesh(f"zg_ablation/real_zg/sample_{i}",
                        vertices=real_cpu[i].unsqueeze(0),
                        colors=blue.unsqueeze(0), global_step=epoch)
        writer.add_mesh(f"zg_ablation/random_zg/sample_{i}",
                        vertices=rand_cpu[i].unsqueeze(0),
                        colors=orange.unsqueeze(0), global_step=epoch)

    model.train()


@torch.no_grad()
def log_style_samples(writer, model, device, epoch,
                      n_fixed: int = 3, n_styles: int = 5, seed: int = 0):
    """
    Fix n_fixed z_l vectors, decode each with n_styles fresh z_g samples.
    Shows how z_g controls appearance for a fixed shape code.
    """
    model.eval()
    gen = torch.Generator()
    gen.manual_seed(seed)
    fixed_zl = torch.randn(n_fixed, model.latent_size,
                           generator=gen).clamp(-2, 2).to(device)
    blue = torch.tensor([[60, 120, 220]], dtype=torch.uint8)

    for i in range(n_fixed):
        zl = fixed_zl[i].unsqueeze(0)     # (1, latent_size)
        for j in range(n_styles):
            zg  = torch.randn(1, model.style_dim, device=device)
            pts = model.decoder(zl, zg)   # (1, N, 3)
            pts = pts.detach().float().cpu().squeeze(0)
            clr = blue.expand(pts.shape[0], -1).unsqueeze(0)
            writer.add_mesh(f"samples/latent_{i}/style_{j}",
                            vertices=pts.unsqueeze(0),
                            colors=clr, global_step=epoch)
    model.train()


@torch.no_grad()
def log_latent_analysis(writer, model, loader, device, epoch):
    """
    Per-dimension KL for z_l (flat) and z_g.
    active_units = dims with mean KL > 0.1.
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)

    _, mu_l, logvar_l, mu_g, logvar_g = model(points)

    def kl_per_dim(mu, logvar):
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean(dim=0)   # (D,)

    kl_shape  = kl_per_dim(mu_l, logvar_l)   # (latent_size,)
    kl_global = kl_per_dim(mu_g, logvar_g)   # (style_dim,)

    thr = 0.1
    writer.add_scalar("latent/active_units_shape",  (kl_shape  > thr).sum().item(), epoch)
    writer.add_scalar("latent/active_units_global", (kl_global > thr).sum().item(), epoch)
    writer.add_scalar("latent/mean_kl_shape",  kl_shape.mean().item(),  epoch)
    writer.add_scalar("latent/mean_kl_global", kl_global.mean().item(), epoch)

    if torch.isfinite(kl_shape).all():
        writer.add_histogram("latent/kl_per_dim_shape",  kl_shape,  epoch)
    if torch.isfinite(kl_global).all():
        writer.add_histogram("latent/kl_per_dim_global", kl_global, epoch)

    model.train()


def log_gradient_norms(writer, model, step: int):
    modules = {
        "style_encoder": model.style_encoder,
        "shape_encoder": model.shape_encoder,
        "decoder":       model.decoder,
    }
    total_sq = 0.0
    for name, mod in modules.items():
        sq = sum(p.grad.detach().norm(2).item() ** 2
                 for p in mod.parameters() if p.grad is not None)
        writer.add_scalar(f"gradients/{name}", sq ** 0.5, step)
        total_sq += sq
    writer.add_scalar("gradients/total_before_clip", total_sq ** 0.5, step)


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

    for batch_idx, (points, _) in enumerate(loader):
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, mu_l, logvar_l, mu_g, logvar_g = model(points)

            recon, cd_f, cd_b = chamfer_distance_knn(xyz_out, target_xyz)
            kl_shape = kl_divergence(mu_l, logvar_l)
            kl_style = kl_divergence(mu_g, logvar_g)
            total    = (recon
                        + beta * cfg.beta_shape * kl_shape
                        + beta * cfg.beta_style * kl_style)

        losses = {
            "total":    total,
            "recon":    recon,
            "kl_shape": kl_shape,
            "kl_style": kl_style,
            "cd_fwd":   cd_f,
            "cd_bwd":   cd_b,
        }

        if not torch.isfinite(total):
            optimiser.zero_grad(set_to_none=True)
            logger.warning(f"  Epoch {epoch:03d}  Batch {batch_idx+1}: loss não-finita, ignorado")
            continue

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimiser)

        if writer is not None and (batch_idx + 1) % cfg.log_every == 0:
            log_gradient_norms(writer, model, epoch * len(loader) + batch_idx)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            optimiser.zero_grad(set_to_none=True)
            scaler.update()
            logger.warning(f"  Epoch {epoch:03d}  Batch {batch_idx+1}: grad não-finito, ignorado")
            continue

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
            xyz_out, mu_l, logvar_l, mu_g, logvar_g = model(points)

            recon, cd_f, cd_b = chamfer_distance_knn(xyz_out, target_xyz)
            kl_shape = kl_divergence(mu_l, logvar_l)
            kl_style = kl_divergence(mu_g, logvar_g)
            total    = (recon
                        + beta * cfg.beta_shape * kl_shape
                        + beta * cfg.beta_style * kl_style)

        fs = f_score(xyz_out, target_xyz, threshold=0.01)

        for k, v in {"total": total, "recon": recon,
                     "kl_shape": kl_shape, "kl_style": kl_style}.items():
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
        latent_size=cfg.latent_size,
        style_dim=cfg.style_dim,
        in_channels=cfg.in_channels,
        num_points=cfg.num_points,
    ).to(device)
    logger.info(f"Flat VAE  |  params: {count_parameters(model):,}")
    logger.info(f"  latent_size={cfg.latent_size}  style_dim={cfg.style_dim}")

    # ---- Optimiser & scheduler ------------------------------------------
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup    = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine    = CosineAnnealingLR(optimiser, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume ---------------------------------------------------------
    start_epoch = 0
    best_val_cd = math.inf
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
            model, trn_loader, optimiser, scaler, cfg, epoch, logger, device, writer=writer)
        val_metrics = validate(model, val_loader, cfg, epoch, device)
        scheduler.step()

        elapsed = time.time() - t0
        beta    = beta_schedule(epoch, cfg)
        trn_str = "  ".join(f"trn_{k}={v:.5f}" for k, v in trn_metrics.items())
        val_str = "  ".join(f"val_{k}={v:.5f}" for k, v in val_metrics.items())
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  β={beta:.4f}  time={elapsed:.1f}s\n"
            f"  {trn_str}\n  {val_str}"
        )

        writer.add_scalar("hparams/lr",   scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("hparams/beta", beta,                        epoch)
        log_scalars(writer, trn_metrics, "train", epoch)
        log_scalars(writer, val_metrics, "val",   epoch)

        if epoch % 5 == 0:
            log_latent_analysis(writer, model, val_loader, device, epoch)
            log_gt_recon(writer, model, trn_loader, device, epoch, "train")
            log_gt_recon(writer, model, val_loader,  device, epoch, "val")
            log_zg_ablation(writer, model, val_loader, device, epoch)
            log_style_samples(writer, model, device, epoch)

        val_cd  = val_metrics.get("recon", math.inf)
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
    p   = argparse.ArgumentParser(description="Train Flat Point-Cloud VAE")
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
