"""
train_diffusion.py
------------------
Train the two-stage diffusion on top of the flat two-vector VAE (Real_latent).

  Stage 1  StyleDenoiser  :  class label → z_g           (global style DDPM)
  Stage 2  FlatDenoiser   :  z_g         → z_l ∈ ℝ^L    (flat shape DDPM)

Both stages are trained simultaneously per batch:
  1. Run frozen VAE encoders → clean latents (posterior mean, no noise).
  2. Sample random timesteps → add noise.
  3. Each denoiser predicts the noise → MSE loss.
  4. Backprop only through denoisers (VAE is frozen).

Generation (at inference / TensorBoard):
  1. Sample z_g from StyleDenoiser (class-conditioned, DDPM chain).
  2. Sample z_l from FlatDenoiser  (z_g-conditioned, DDPM chain).
  3. Decode via frozen VAE decoder → point cloud.
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
from src.Vae import Vae, normalize_pc
from src.Diffusion import LinearSchedule, StyleDenoiser, FlatDenoiser
from src.metric import generation_metrics


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

    # --- VAE architecture (auto-read from checkpoint; these are fallback defaults) ---
    vae_latent_size: int = 1024
    vae_style_dim:   int = 128
    vae_in_channels: int = 6
    vae_num_points:  int = 2048

    # --- diffusion schedule ---
    T:           int   = 1000

    # --- Style denoiser (Stage 1: class → z_g) ---
    num_classes:    int   = 55
    style_hidden:   int   = 512
    style_layers:   int   = 6
    cfg_dropout:    float = 0.1
    guidance:       float = 3.0

    # --- Flat shape denoiser (Stage 2: z_g → z_l) ---
    shape_hidden:   int   = 512
    shape_layers:   int   = 8

    # --- training ---
    epochs:         int   = 300
    batch_size:     int   = 16
    lr:             float = 2e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0
    ema_decay:      float = 0.9999

    # --- data ---
    val_split:   float = 0.1
    num_workers: int   = 4
    pin_memory:  bool  = True

    # --- misc ---
    seed:        int   = 42
    save_every:  int   = 10
    log_every:   int   = 50
    device:      str   = "cuda"
    amp:         bool  = False
    resume:      int   = 0

    # --- generation visualization ---
    vis_every:      int   = 10
    vis_per_class:  int   = 2

    # --- generation metrics (MMD / COV / 1-NNA) ---
    gen_metrics_every: int = 25
    gen_metrics_n:     int = 512


# ============================================================
# Latent extraction (frozen VAE)
# ============================================================

@torch.no_grad()
def extract_latents(vae: Vae, points: torch.Tensor, stats: dict | None = None):
    """
    Run the frozen VAE encoders and return clean latents (posterior mean).

    Returns
    -------
    z_g  (B, style_dim)    global style posterior mean
    z_l  (B, latent_size)  flat shape posterior mean
    """
    x_norm  = normalize_pc(points)
    mu_g, _ = vae.style_encoder(x_norm)   # (B, style_dim)
    mu_l, _ = vae.shape_encoder(x_norm)   # (B, latent_size)

    if stats is not None:
        mu_g = (mu_g - stats["zg_mean"]) / stats["zg_std"]
        mu_l = (mu_l - stats["zl_mean"]) / stats["zl_std"]
    return mu_g, mu_l


@torch.no_grad()
def compute_latent_stats(vae: Vae, loader: DataLoader, device: torch.device) -> dict:
    """Per-channel mean/std of latents over the training set."""
    zg_sum = zg_sq = None; zg_n = 0
    zl_sum = zl_sq = None; zl_n = 0
    for points, _ in loader:
        points = points.to(device)
        zg, zl = extract_latents(vae, points)    # raw (B, style_dim), (B, latent_size)
        zg_sum = zg.sum(0)       if zg_sum is None else zg_sum + zg.sum(0)
        zg_sq  = (zg*zg).sum(0) if zg_sq  is None else zg_sq  + (zg*zg).sum(0)
        zg_n  += zg.shape[0]
        zl_sum = zl.sum(0)       if zl_sum is None else zl_sum + zl.sum(0)
        zl_sq  = (zl*zl).sum(0) if zl_sq  is None else zl_sq  + (zl*zl).sum(0)
        zl_n  += zl.shape[0]
    zg_mean = zg_sum / zg_n
    zg_std  = (zg_sq / zg_n - zg_mean**2).clamp(min=1e-4).sqrt().clamp(min=1e-2)
    zl_mean = zl_sum / zl_n
    zl_std  = (zl_sq / zl_n - zl_mean**2).clamp(min=1e-4).sqrt().clamp(min=1e-2)
    return {"zg_mean": zg_mean, "zg_std": zg_std, "zl_mean": zl_mean, "zl_std": zl_std}


def denorm_zg(z_g: torch.Tensor, stats: dict | None) -> torch.Tensor:
    return z_g if stats is None else z_g * stats["zg_std"] + stats["zg_mean"]


def denorm_zl(z_l: torch.Tensor, stats: dict | None) -> torch.Tensor:
    return z_l if stats is None else z_l * stats["zl_std"] + stats["zl_mean"]


# ============================================================
# Loss
# ============================================================

def diffusion_loss(
    schedule:  LinearSchedule,
    denoiser:  nn.Module,
    x0:        torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    B  = x0.shape[0]
    t  = torch.randint(0, schedule.T, (B,), device=x0.device)
    xt, noise = schedule.q_sample(x0, t)
    pred      = denoiser(xt, t, condition)
    return F.mse_loss(pred, noise)


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


@torch.no_grad()
def log_generations(
    writer:      SummaryWriter,
    schedule:    LinearSchedule,
    style_dn:    StyleDenoiser,
    shape_dn:    FlatDenoiser,
    vae:         Vae,
    cfg:         DiffusionConfig,
    epoch:       int,
    device:      torch.device,
    class_names: dict,
    stats=None,
):
    style_dn.eval(); shape_dn.eval(); vae.eval()
    show_classes = list(range(min(4, cfg.num_classes)))
    latent_size  = vae.latent_size

    for cls_idx in show_classes:
        B   = cfg.vis_per_class
        cls = torch.full((B,), cls_idx, device=device, dtype=torch.long)

        uncond = style_dn.uncond(B, device)
        z_g    = schedule.sample(
            style_dn, (vae.style_dim,),
            condition=cls, uncond=uncond,
            guidance=cfg.guidance, device=device,
        )
        z_l    = schedule.sample(
            shape_dn, (latent_size,),
            condition=z_g, device=device,
        )

        xyz_out = vae.decoder(denorm_zl(z_l, stats), denorm_zg(z_g, stats))
        xyz_out = xyz_out.float().cpu()

        N = xyz_out.shape[1]
        for i in range(B):
            gen_v = xyz_out[i]
            gen_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(N, -1)
            verts, clrs = _side_by_side([gen_v], [gen_c])
            cls_name    = class_names.get(cls_idx, str(cls_idx))
            writer.add_mesh(
                f"generated/{cls_name}_sample{i}",
                vertices=verts, colors=clrs, global_step=epoch,
            )

    style_dn.train(); shape_dn.train()


# ============================================================
# Train / Val steps
# ============================================================

def train_one_epoch(
    vae:       Vae,
    schedule:  LinearSchedule,
    style_dn:  StyleDenoiser,
    shape_dn:  FlatDenoiser,
    loader:    DataLoader,
    opt_s:     torch.optim.Optimizer,
    opt_p:     torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    cfg:       DiffusionConfig,
    epoch:     int,
    logger:    logging.Logger,
    device:    torch.device,
    ema_style=None,
    ema_shape=None,
    stats=None,
) -> dict:
    style_dn.train(); shape_dn.train()
    totals    = {}
    n_batches = 0

    for batch_idx, (points, class_idx) in enumerate(loader):
        points    = points.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        z_g, z_l = extract_latents(vae, points, stats)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx)

        opt_s.zero_grad(set_to_none=True)
        scaler.scale(loss_s).backward()
        scaler.unscale_(opt_s)
        nn.utils.clip_grad_norm_(style_dn.parameters(), cfg.grad_clip)
        scaler.step(opt_s)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_p = diffusion_loss(schedule, shape_dn, z_l, z_g)

        opt_p.zero_grad(set_to_none=True)
        scaler.scale(loss_p).backward()
        scaler.unscale_(opt_p)
        nn.utils.clip_grad_norm_(shape_dn.parameters(), cfg.grad_clip)
        scaler.step(opt_p)

        scaler.update()

        if ema_style is not None:
            ema_style.update_parameters(style_dn)
        if ema_shape is not None:
            ema_shape.update_parameters(shape_dn)

        totals["loss_style"] = totals.get("loss_style", 0.0) + loss_s.item()
        totals["loss_shape"] = totals.get("loss_shape", 0.0) + loss_p.item()
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
    vae:       Vae,
    schedule:  LinearSchedule,
    style_dn:  StyleDenoiser,
    shape_dn:  FlatDenoiser,
    loader:    DataLoader,
    cfg:       DiffusionConfig,
    device:    torch.device,
    compute_gen_metrics: bool = False,
    ema_style=None,
    ema_shape=None,
    stats=None,
) -> dict:
    style_dn.eval(); shape_dn.eval()
    totals    = {}
    n_batches = 0
    ref_clouds: list[torch.Tensor] = []

    for points, class_idx in loader:
        points    = points.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        z_g, z_l = extract_latents(vae, points, stats)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx)
            loss_p = diffusion_loss(schedule, shape_dn, z_l, z_g)

        totals["loss_style"] = totals.get("loss_style", 0.0) + loss_s.item()
        totals["loss_shape"] = totals.get("loss_shape", 0.0) + loss_p.item()
        n_batches += 1

        if compute_gen_metrics:
            ref_clouds.append(normalize_pc(points)[..., :3].float().cpu())

    out = {k: v / max(n_batches, 1) for k, v in totals.items()}

    if compute_gen_metrics:
        gen_style = ema_style.module if ema_style is not None else style_dn
        gen_shape = ema_shape.module if ema_shape is not None else shape_dn
        out.update(_eval_generation(
            schedule, gen_style, gen_shape, vae, ref_clouds, cfg, device, stats=stats
        ))

    return out


@torch.no_grad()
def _eval_generation(
    schedule:   LinearSchedule,
    style_dn:   StyleDenoiser,
    shape_dn:   FlatDenoiser,
    vae:        Vae,
    ref_clouds: list[torch.Tensor],
    cfg:        DiffusionConfig,
    device:     torch.device,
    n_gen:      int = 512,
    stats=None,
) -> dict:
    latent_size = vae.latent_size
    gen_clouds  = []

    for start in range(0, n_gen, cfg.batch_size):
        B   = min(cfg.batch_size, n_gen - start)
        cls = torch.randint(0, cfg.num_classes, (B,), device=device)

        uncond = style_dn.uncond(B, device)
        z_g    = schedule.sample(style_dn, (vae.style_dim,),
                                 condition=cls, uncond=uncond,
                                 guidance=cfg.guidance, device=device)
        z_l    = schedule.sample(shape_dn, (latent_size,),
                                 condition=z_g, device=device)
        xyz    = vae.decoder(denorm_zl(z_l, stats), denorm_zg(z_g, stats))
        gen_clouds.append(xyz.float().cpu())

    gen = torch.cat(gen_clouds, dim=0)
    ref = torch.cat(ref_clouds, dim=0)

    if len(ref) > n_gen:
        idx = torch.randperm(len(ref))[:n_gen]
        ref = ref[idx]

    metrics = generation_metrics(gen.to(device), ref.to(device))
    return {f"gen_{k}": v for k, v in metrics.items()}


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
    if not os.path.exists(cfg.vae_ckpt):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {cfg.vae_ckpt}\n"
            "Run train_vae.py first."
        )
    ckpt = torch.load(cfg.vae_ckpt, map_location=device, weights_only=False)

    saved_cfg = ckpt.get("config", {})
    vae_latent_size  = saved_cfg.get("latent_size",  cfg.vae_latent_size)
    vae_style_dim    = saved_cfg.get("style_dim",    cfg.vae_style_dim)
    vae_in_channels  = saved_cfg.get("in_channels",  cfg.vae_in_channels)
    vae_num_points   = saved_cfg.get("num_points",   cfg.vae_num_points)
    logger.info(
        f"VAE arch from checkpoint: latent_size={vae_latent_size} "
        f"style_dim={vae_style_dim} in_channels={vae_in_channels} "
        f"num_points={vae_num_points}"
    )

    vae = Vae(
        latent_size=vae_latent_size,
        style_dim=vae_style_dim,
        in_channels=vae_in_channels,
        num_points=vae_num_points,
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"VAE loaded from {cfg.vae_ckpt}  (frozen)")

    from src.dataset import Ds_point_model
    idx_to_name = {idx: name for idx, (_, name) in
                   enumerate(Ds_point_model.map().items())}

    # ---- Diffusion models -----------------------------------------
    schedule  = LinearSchedule(T=cfg.T).to(device)

    style_dn  = StyleDenoiser(
        style_dim=vae_style_dim,
        num_classes=cfg.num_classes,
        hidden=cfg.style_hidden,
        n_layers=cfg.style_layers,
        T=cfg.T,
        cfg_dropout=cfg.cfg_dropout,
    ).to(device)

    shape_dn  = FlatDenoiser(
        latent_size=vae_latent_size,
        style_dim=vae_style_dim,
        hidden=cfg.shape_hidden,
        n_layers=cfg.shape_layers,
        T=cfg.T,
    ).to(device)

    logger.info(f"StyleDenoiser  params: {count_parameters(style_dn):,}")
    logger.info(f"FlatDenoiser   params: {count_parameters(shape_dn):,}")

    from torch.optim.swa_utils import AveragedModel

    def make_ema_avg(base_decay: float):
        def ema_avg(avg_p, p, num_averaged):
            n = float(num_averaged.item() if hasattr(num_averaged, "item") else num_averaged)
            d = min(base_decay, (1.0 + n) / (10.0 + n))
            return avg_p * d + p * (1.0 - d)
        return ema_avg

    ema_style = AveragedModel(style_dn, avg_fn=make_ema_avg(cfg.ema_decay))
    ema_shape = AveragedModel(shape_dn, avg_fn=make_ema_avg(cfg.ema_decay))

    # ---- Data ---------------------------------------------------------
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices = torch.randperm(len(base_ds),
                             generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n   = max(1, int(len(base_ds) * cfg.val_split))
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    trn_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=True), train_idx)
    val_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)

    trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val")

    # ---- Latent standardisation -----------------------------------
    stats = compute_latent_stats(vae, trn_loader, device)
    logger.info(
        f"Latent stats | z_g: ||mean||={stats['zg_mean'].norm():.2f} "
        f"std∈[{stats['zg_std'].min():.3f},{stats['zg_std'].max():.3f}] | "
        f"z_l: std_mean={stats['zl_std'].mean():.3f}"
    )

    # ---- Optimisers -------------------------------------------------
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
    opt_p, sched_p = make_opt_sched(shape_dn)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume ---------------------------------------------------
    start_epoch    = 0
    best_val_loss  = math.inf
    history: list  = []

    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt_d = torch.load(resume_path, map_location=device, weights_only=False)
        style_dn.load_state_dict(ckpt_d["style_dn"])
        shape_dn.load_state_dict(ckpt_d["shape_dn"])
        if "ema_style" in ckpt_d:
            ema_style.load_state_dict(ckpt_d["ema_style"])
        if "ema_shape" in ckpt_d:
            ema_shape.load_state_dict(ckpt_d["ema_shape"])
        opt_s.load_state_dict(ckpt_d["opt_s"])
        opt_p.load_state_dict(ckpt_d["opt_p"])
        sched_s.load_state_dict(ckpt_d["sched_s"])
        sched_p.load_state_dict(ckpt_d["sched_p"])
        scaler.load_state_dict(ckpt_d["scaler"])
        start_epoch   = ckpt_d["epoch"] + 1
        best_val_loss = ckpt_d.get("best_val_loss", math.inf)
        history       = ckpt_d.get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop --------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn = train_one_epoch(
            vae, schedule, style_dn, shape_dn,
            trn_loader, opt_s, opt_p, scaler, cfg, epoch, logger, device,
            ema_style=ema_style, ema_shape=ema_shape, stats=stats,
        )
        compute_gen = (cfg.gen_metrics_every > 0 and epoch % cfg.gen_metrics_every == 0)
        val = validate(vae, schedule, style_dn, shape_dn, val_loader, cfg, device,
                       compute_gen_metrics=compute_gen,
                       ema_style=ema_style, ema_shape=ema_shape, stats=stats)

        sched_s.step(); sched_p.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={sched_s.get_last_lr()[0]:.2e}  time={elapsed:.1f}s\n"
            f"  trn: {trn}\n  val: {val}"
        )

        writer.add_scalar("train/lr_style", sched_s.get_last_lr()[0], epoch)
        writer.add_scalar("train/lr_shape", sched_p.get_last_lr()[0], epoch)
        log_metrics(writer, trn, "train", epoch)
        log_metrics(writer, val, "val",   epoch)

        if epoch % cfg.vis_every == 0:
            log_generations(
                writer, schedule, ema_style.module, ema_shape.module, vae, cfg, epoch,
                device, idx_to_name, stats=stats,
            )

        val_loss = val.get("loss_style", math.inf) + val.get("loss_shape", math.inf)
        is_best  = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"  New best val loss: {best_val_loss:.5f}")

        save_state = {
            "epoch":         epoch,
            "style_dn":      style_dn.state_dict(),
            "shape_dn":      shape_dn.state_dict(),
            "ema_style":     ema_style.state_dict(),
            "ema_shape":     ema_shape.state_dict(),
            "opt_s":         opt_s.state_dict(),
            "opt_p":         opt_p.state_dict(),
            "sched_s":       sched_s.state_dict(),
            "sched_p":       sched_p.state_dict(),
            "scaler":        scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "config":        asdict(cfg),
            "history":       history,
            "latent_stats":  {k: v.detach().cpu() for k, v in stats.items()},
        }
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))
        if is_best:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))

        history.append({"epoch": epoch, **{f"trn_{k}": v for k, v in trn.items()},
                                          **{f"val_{k}": v for k, v in val.items()}})
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> DiffusionConfig:
    cfg = DiffusionConfig()
    p   = argparse.ArgumentParser(description="Train two-stage flat diffusion")
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
