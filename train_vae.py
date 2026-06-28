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
from src.metric import chamfer_distance_knn, f_score, vae_loss
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
    latent_dim:    int   = 8       # per-point latent features
    style_dim:     int   = 128     # global style vector (LION paper: 128)
    in_channels:   int   = 6

    # --- training ---
    epochs:        int   = 200
    batch_size:    int   = 16
    lr:            float = 1e-4    # LION: learning_rate_vae = 1e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0

    # --- KL annealing ---
    beta_start:    float = 1e-7
    beta_end:      float = 0.1
    beta_epochs:   int   = 150
    # Per-latent KL weights (mirrors LION weight_kl_pt / weight_kl_feat / weight_kl_glb).
    # All 1.0 → uniform regularisation, matching LION's design intent.
    beta_style:    float = 1.0
    beta_pos_pts:  float = 1.0
    beta_feat_pts: float = 1.0

    # Position noise: breaks the z_l position shortcut so decoder uses z_g more.
    pos_noise_std: float = 0.05

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
    """Log a dict of scalars under losses/{group}/."""
    for k, v in metrics.items():
        writer.add_scalar(f"losses/{group}/{k}", v, epoch)


# ---- GT vs Reconstruction ---------------------------------------------------

@torch.no_grad()
def log_gt_recon(writer, model, loader, device, epoch, tag: str, max_items: int = 4):
    """
    Green = GT,  Red = reconstruction (posterior means, no noise).

    Tags:  recon/{tag}/gt/sample_{i}
           recon/{tag}/recon/sample_{i}
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)
    N = points.shape[1]

    xyz_out, _ = model.reconstruct(points)
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


# ---- z_g ablation -----------------------------------------------------------

@torch.no_grad()
def log_zg_ablation(writer, model, loader, device, epoch, max_items: int = 4):
    """
    Diagnostic: does z_g actually influence the output?

    Same z_l for all columns, only z_g changes:
      GREEN  = GT
      BLUE   = decode with real z_g  (posterior mean)
      ORANGE = decode with random z_g ~ N(0,I)

    If blue ≈ orange → decoder ignores z_g → posterior collapse.
    If blue ≠ orange → z_g carries information → healthy.

    Scalars logged:
      zg_ablation/cd_real_zg    — CD with real z_g (should be low)
      zg_ablation/cd_random_zg  — CD with random z_g
      zg_ablation/cd_ratio      — random / real  (1.0 = z_g useless, >1 = z_g helps)
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points[:max_items].to(device)

    x_norm     = normalize_pc(points)
    target_xyz = x_norm[..., :3]
    xyz_anchor = x_norm[..., :3]

    # Posterior means — deterministic, no noise
    mu_g, _  = model.global_encoder(x_norm)
    mu_l, _  = model.local_encoder(x_norm, mu_g)

    # z_l with position skip (same as model.reconstruct)
    z_l = mu_l.clone()
    z_l[..., :3] = xyz_anchor + model.skip_weight * mu_l[..., :3]

    # Decode with real z_g
    xyz_real, _ = model.decoder(z_l, mu_g)

    # Decode with random z_g — same z_l, different z_g
    z_g_rand    = torch.randn_like(mu_g)
    xyz_rand, _ = model.decoder(z_l, z_g_rand)

    # CD comparison — chamfer_distance_knn returns (cd, cd_fwd, cd_bwd)
    cd_real = chamfer_distance_knn(xyz_real, target_xyz, reduce="mean")[0].item()
    cd_rand = chamfer_distance_knn(xyz_rand, target_xyz, reduce="mean")[0].item()
    ratio   = cd_rand / (cd_real + 1e-8)

    writer.add_scalar("zg_ablation/cd_real_zg",   cd_real, epoch)
    writer.add_scalar("zg_ablation/cd_random_zg", cd_rand, epoch)
    writer.add_scalar("zg_ablation/cd_ratio",     ratio,   epoch)

    # Point cloud meshes
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


# ---- Style samples (fixed z_l, varying z_g) ---------------------------------

@torch.no_grad()
def log_style_samples(writer, model, device, epoch, num_points: int,
                      n_fixed: int = 3, n_styles: int = 5, seed: int = 0):
    """
    Fix n_fixed z_l vectors (same across epochs via seed), decode each with
    n_styles fresh z_g samples. Shows how z_g controls shape for a fixed local.

    Tags: samples/latent_{i}/style_{j}
    """
    model.eval()
    gen = torch.Generator()
    gen.manual_seed(seed)
    fixed_zl = torch.randn(n_fixed, num_points, model.total_z_dim,
                           generator=gen).clamp(-1, 1).to(device)
    blue = torch.tensor([[60, 120, 220]], dtype=torch.uint8)

    for i in range(n_fixed):
        zl = fixed_zl[i].unsqueeze(0)
        for j in range(n_styles):
            zg       = torch.randn(1, model.style_dim, device=device)
            pts, _   = model.decoder(zl, zg)
            pts      = pts.detach().float().cpu().squeeze(0)
            clr      = blue.expand(pts.shape[0], -1).unsqueeze(0)
            writer.add_mesh(f"samples/latent_{i}/style_{j}",
                            vertices=pts.unsqueeze(0),
                            colors=clr, global_step=epoch)
    model.train()


# ---- Latent analysis --------------------------------------------------------

@torch.no_grad()
def log_latent_analysis(writer, model, loader, device, epoch):
    """
    Per-dimension KL for z_l and z_g.

    active_units = dims with mean KL > 0.1.
    If active_units_local == latent_dim → all dims used.
    If active_units_global << style_dim → z_g is collapsing.
    Histograms show the full per-dim KL distribution at a glance.
    """
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)

    _, _, mu_l, logvar_l, mu_g, logvar_g = model(points)

    def kl_per_dim(mu, logvar):
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        while kl.dim() > 1:
            kl = kl.mean(dim=0)
        return kl

    kl_local  = kl_per_dim(mu_l,  logvar_l)   # (latent_dim,)
    kl_global = kl_per_dim(mu_g,  logvar_g)   # (style_dim,)

    thr = 0.1
    writer.add_scalar("latent/active_units_local",  (kl_local  > thr).sum().item(), epoch)
    writer.add_scalar("latent/active_units_global", (kl_global > thr).sum().item(), epoch)
    writer.add_scalar("latent/mean_kl_local",  kl_local.mean().item(),  epoch)
    writer.add_scalar("latent/mean_kl_global", kl_global.mean().item(), epoch)

    if torch.isfinite(kl_local).all():
        writer.add_histogram("latent/kl_per_dim_local",  kl_local,  epoch)
    if torch.isfinite(kl_global).all():
        writer.add_histogram("latent/kl_per_dim_global", kl_global, epoch)

    model.train()


# ---- Gradient norms ---------------------------------------------------------

def log_gradient_norms(writer, model, step: int):
    modules = {
        "global_encoder": model.global_encoder,
        "local_encoder":  model.local_encoder,
        "decoder":        model.decoder,
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
            xyz_out, _, mu_l, logvar_l, mu_g, logvar_g = model(
                points, pos_noise_std=cfg.pos_noise_std)
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
            )

        if not torch.isfinite(losses["total"]):
            optimiser.zero_grad(set_to_none=True)
            logger.warning(f"  Epoch {epoch:03d}  Batch {batch_idx+1}: loss não-finita, ignorado")
            continue

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
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
            xyz_out, _, mu_l, logvar_l, mu_g, logvar_g = model(
                points, pos_noise_std=cfg.pos_noise_std)
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

    sample_pts, _ = trn_ds[0]
    num_points = sample_pts.shape[0]

    # ---- Model ----------------------------------------------------------
    model = Vae(
        latent_dim=cfg.latent_dim,
        style_dim=cfg.style_dim,
        in_channels=cfg.in_channels,
    ).to(device)
    logger.info(f"LION VAE  |  params: {count_parameters(model):,}")
    logger.info(f"  latent_dim={cfg.latent_dim}  style_dim={cfg.style_dim}  total_z_dim={model.total_z_dim}")

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

        # --- TensorBoard scalars ---
        writer.add_scalar("hparams/lr",   scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("hparams/beta", beta,                        epoch)
        log_scalars(writer, trn_metrics, "train", epoch)
        log_scalars(writer, val_metrics, "val",   epoch)

        # --- Periodic visualisations (every 5 epochs) ---
        if epoch % 5 == 0:
            # Latent space diagnostics
            log_latent_analysis(writer, model, val_loader, device, epoch)

            # GT vs reconstruction
            log_gt_recon(writer, model, trn_loader, device, epoch, "train")
            log_gt_recon(writer, model, val_loader,  device, epoch, "val")

            # z_g ablation: real vs random z_g, same z_l
            log_zg_ablation(writer, model, val_loader, device, epoch)

            # Style diversity: fixed z_l, varying z_g
            log_style_samples(writer, model, device, epoch, num_points=num_points)

        # --- Checkpointing ---
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
