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

    # --- VAE architecture (must match the loaded checkpoint) ---
    # Which class to instantiate (Vae vs VaeUp) and its extra kwargs (n_latent, n_points, k)
    # are auto-detected from the checkpoint's saved config — no CLI flag needed for that.
    vae_latent_dim:  int = 8   # must match the latent_dim used in train_vae.py / train_vae_up.py
    vae_style_dim:   int = 256
    vae_in_channels: int = 6

    # --- number of points in z_local ---
    # Vae: variable per-sample (from the dataset), only used here for the vis/generation shape.
    # VaeUp: MUST equal the checkpoint's n_latent (auto-verified against ckpt config below).
    n_points:  int = 2048

    # --- diffusion schedule ---
    T:           int   = 1000

    # --- Style denoiser (Stage 1) ---
    num_classes:    int   = 55
    style_hidden:   int   = 512
    style_layers:   int   = 6
    cfg_dropout:    float = 0.1
    guidance:       float = 3.0    # CFG scale at inference

    # --- Latent point denoiser (Stage 2) ---
    point_hidden:   int   = 256
    point_layers:   int   = 8

    # --- training ---
    epochs:         int   = 300
    batch_size:     int   = 16
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0

    # --- data ---
    val_split:   float = 0.1
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

    # --- generation-quality metrics (MMD / COV / 1-NNA, LION protocol) ---
    # Full DDPM sampling is expensive, so this runs on the same cadence as
    # visualization rather than every epoch, and is used to pick a second
    # checkpoint (best_gen.pt) that reflects generation quality directly,
    # since the denoising MSE loss (used for best.pt) is only a proxy for it.
    eval_gen_metrics:       bool  = True
    eval_every:             int   = 10    # compute MMD/COV/1-NNA every N epochs
    eval_samples_per_class: int   = 16    # generated shapes per class
    eval_max_ref:           int   = 32    # reference (real) shapes per class, capped
    eval_metric:            str   = "cd"  # "cd" (Chamfer) or "emd"
    eval_batch_size:        int   = 16    # inner batch size for the pairwise distance matrix


# ============================================================
# Latent extraction (frozen VAE)
# ============================================================

@torch.no_grad()
def extract_latents(vae: Vae, points: torch.Tensor):
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


# ============================================================
# Loss
# ============================================================

def diffusion_loss(
    schedule:   CosineSchedule,
    denoiser:   nn.Module,
    x0:         torch.Tensor,
    condition:  torch.Tensor,
) -> torch.Tensor:
    """Simple DDPM ε-prediction MSE loss."""
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
    writer:         SummaryWriter,
    schedule:       CosineSchedule,
    style_dn:       StyleDenoiser,
    point_dn:       LatentPointDenoiser,
    vae:            Vae,
    cfg:            DiffusionConfig,
    epoch:          int,
    device:         torch.device,
    class_names:    dict,          # idx → name string
    show_classes:   list,          # class indices actually present in the training data
):
    """
    For a handful of classes, generate shapes via the full two-stage chain
    and log them to TensorBoard alongside a real sample from that class.
    """
    style_dn.eval(); point_dn.eval(); vae.eval()

    N  = cfg.n_points   # number of latent points to sample (Vae: full N, VaeUp: n_latent)
    pd = cfg.vae_latent_dim

    for cls_idx in show_classes:
        B   = cfg.vis_per_class
        cls = torch.full((B,), cls_idx, device=device, dtype=torch.long)

        # --- Stage 1: sample z_g ---
        uncond = style_dn.uncond(B, device)
        z_g    = schedule.sample(
            style_dn, (cfg.vae_style_dim,),
            condition=cls, uncond=uncond,
            guidance=cfg.guidance, device=device,
        )   # (B, style_dim)

        # --- Stage 2: sample z_local conditioned on z_g ---
        z_local = schedule.sample(
            point_dn, (N, pd),
            condition=z_g, device=device,
        )   # (B, N, 6)

        # --- Decode ---
        xyz_out = vae.decoder(z_local, z_g).float().cpu()  # (B, N, 3)  z_local is z_l here

        for i in range(B):
            gen_v = xyz_out[i]
            gen_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(N, -1)
            verts, clrs = _side_by_side([gen_v], [gen_c])
            cls_name    = class_names.get(cls_idx, str(cls_idx))
            writer.add_mesh(
                f"generated/{cls_name}_sample{i}",
                vertices=verts, colors=clrs, global_step=epoch,
            )

    style_dn.train(); point_dn.train()


@torch.no_grad()
def evaluate_generation_metrics(
    schedule:     CosineSchedule,
    style_dn:     StyleDenoiser,
    point_dn:     LatentPointDenoiser,
    vae:          Vae,
    cfg:          DiffusionConfig,
    device:       torch.device,
    show_classes: list,           # class indices actually present in the training data
    ref_pool:     dict,           # class_idx -> list of dataset indices (real shapes)
    ref_dataset,                  # Ds_point_sampled_already (non-augmented), indexable by ref_pool values
) -> dict | None:
    """
    Generate `eval_samples_per_class` shapes per class via the full two-stage
    DDPM chain, compare them against up to `eval_max_ref` real shapes of the
    same class, and average MMD / COV / 1-NNA (src.metric.generation_metrics)
    across classes that have enough reference shapes (>= 2).

    Returns None if no class had enough reference shapes to evaluate.
    """
    style_dn.eval(); point_dn.eval(); vae.eval()

    N  = cfg.n_points
    pd = cfg.vae_latent_dim

    per_class = {}
    for cls_idx in show_classes:
        ref_indices = ref_pool.get(cls_idx, [])[: cfg.eval_max_ref]
        if len(ref_indices) < 2:
            continue

        ref_xyz = torch.stack([ref_dataset[i][0][:, :3] for i in ref_indices]).to(device)
        ref_xyz = normalize_pc(ref_xyz)   # match the frame the VAE decoder outputs

        B   = cfg.eval_samples_per_class
        cls = torch.full((B,), cls_idx, device=device, dtype=torch.long)

        uncond = style_dn.uncond(B, device)
        z_g    = schedule.sample(
            style_dn, (cfg.vae_style_dim,),
            condition=cls, uncond=uncond,
            guidance=cfg.guidance, device=device,
        )
        z_local = schedule.sample(
            point_dn, (N, pd),
            condition=z_g, device=device,
        )
        gen_xyz = vae.decoder(z_local, z_g).float()   # (B, N, 3)

        per_class[cls_idx] = generation_metrics(
            gen_xyz, ref_xyz, metric=cfg.eval_metric, batch_size=cfg.eval_batch_size,
        )

    style_dn.train(); point_dn.train()

    if not per_class:
        return None

    avg = {
        k: sum(m[k] for m in per_class.values()) / len(per_class)
        for k in ("mmd", "cov", "1nna")
    }
    avg["per_class"] = per_class
    return avg


# ============================================================
# Train / Val steps
# ============================================================

def train_one_epoch(
    vae:       Vae,
    schedule:  CosineSchedule,
    style_dn:  StyleDenoiser,
    point_dn:  LatentPointDenoiser,
    loader:    DataLoader,
    opt_s:     torch.optim.Optimizer,
    opt_p:     torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    cfg:       DiffusionConfig,
    epoch:     int,
    logger:    logging.Logger,
    device:    torch.device,
) -> dict:
    style_dn.train(); point_dn.train()
    totals    = {}
    n_batches = 0

    for batch_idx, (points, class_idx) in enumerate(loader):
        points    = points.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        # --- Extract clean latents from frozen VAE ---
        z_g, z_local = extract_latents(vae, points)

        # --- Stage 1: style DDPM loss ---
        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx)

        opt_s.zero_grad(set_to_none=True)
        scaler.scale(loss_s).backward()
        scaler.unscale_(opt_s)
        nn.utils.clip_grad_norm_(style_dn.parameters(), cfg.grad_clip)
        scaler.step(opt_s)

        # --- Stage 2: latent point DDPM loss (condition on z_g mean) ---
        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_p = diffusion_loss(schedule, point_dn, z_local, z_g)

        opt_p.zero_grad(set_to_none=True)
        scaler.scale(loss_p).backward()
        scaler.unscale_(opt_p)
        nn.utils.clip_grad_norm_(point_dn.parameters(), cfg.grad_clip)
        scaler.step(opt_p)

        scaler.update()

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
    vae:       Vae,
    schedule:  CosineSchedule,
    style_dn:  StyleDenoiser,
    point_dn:  LatentPointDenoiser,
    loader:    DataLoader,
    cfg:       DiffusionConfig,
    device:    torch.device,
) -> dict:
    style_dn.eval(); point_dn.eval()
    totals    = {}
    n_batches = 0

    for points, class_idx in loader:
        points    = points.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        z_g, z_local = extract_latents(vae, points)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss_s = diffusion_loss(schedule, style_dn, z_g, class_idx)
            loss_p = diffusion_loss(schedule, point_dn, z_local, z_g)

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
    if not os.path.exists(cfg.vae_ckpt):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {cfg.vae_ckpt}\n"
            "Run train_vae.py (or train_vae_up.py) first and pass --vae_ckpt to this script."
        )
    ckpt     = torch.load(cfg.vae_ckpt, map_location=device)
    vae_cfg  = ckpt.get("config", {})
    is_vae_up = "n_latent" in vae_cfg   # only TrainConfig from train_vae_up.py has this field

    if is_vae_up:
        vae = VaeUp(
            latent_dim=cfg.vae_latent_dim,
            style_dim=cfg.vae_style_dim,
            in_channels=cfg.vae_in_channels,
            n_latent=vae_cfg["n_latent"],
            n_points=vae_cfg["n_points"],
            k=vae_cfg["k"],
        ).to(device)
        if cfg.n_points != vae_cfg["n_latent"]:
            logger.info(
                f"--n_points={cfg.n_points} != checkpoint n_latent={vae_cfg['n_latent']}; "
                f"overriding to match the VaeUp checkpoint."
            )
            cfg.n_points = vae_cfg["n_latent"]
        logger.info(f"Detected VaeUp checkpoint (n_latent={vae_cfg['n_latent']}, "
                    f"n_points={vae_cfg['n_points']}, k={vae_cfg['k']})")
    else:
        vae = Vae(
            latent_dim=cfg.vae_latent_dim,
            style_dim=cfg.vae_style_dim,
            in_channels=cfg.vae_in_channels,
        ).to(device)

    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    logger.info(f"VAE loaded from {cfg.vae_ckpt}  (frozen)")

    # ---- Class name map for logging --------------------------------
    from src.dataset import Ds_point_model
    idx_to_name = {idx: name for idx, (_, name) in
                   enumerate(Ds_point_model.map().items())}

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
        point_dim=cfg.vae_latent_dim,   # no anchor prefix — must match Vae.latent_dim
        style_dim=cfg.vae_style_dim,
        hidden=cfg.point_hidden,
        n_layers=cfg.point_layers,
        T=cfg.T,
    ).to(device)

    logger.info(f"StyleDenoiser      params: {count_parameters(style_dn):,}")
    logger.info(f"LatentPointDenoiser params: {count_parameters(point_dn):,}")

    # ---- Data ---------------------------------------------------------
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices = torch.randperm(len(base_ds),
                             generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n   = max(1, int(len(base_ds) * cfg.val_split))
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    # Class indices are global (position within Ds_point_model.map()'s 51 classes),
    # NOT renumbered for whatever subset of classes you actually train on — e.g.
    # "car" is always index 3 and "airplane" is always index 9, regardless of how
    # many classes are present in cfg.data_root. Detect the classes actually present
    # so visualization doesn't waste slots on untrained classes / miss trained ones.
    show_classes = sorted({
        base_ds.class_to_idx[os.path.basename(f).split("_")[0]] for f in base_ds.files
    })
    logger.info(f"Classes present in data_root: "
                f"{[idx_to_name[c] for c in show_classes]}  (indices {show_classes})")

    trn_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=True), train_idx)
    val_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)

    trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val")

    # class_idx -> list of val-set dataset indices, for the generation-metric
    # reference pool (real shapes to compare generated shapes against).
    ref_pool: dict = {}
    for i in val_idx:
        cls_name = os.path.basename(val_ds.dataset.files[i]).split("_")[0]
        cls_i    = val_ds.dataset.class_to_idx.get(cls_name, 0)
        ref_pool.setdefault(cls_i, []).append(i)

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
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume ---------------------------------------------------
    start_epoch    = 0
    best_val_loss  = math.inf
    best_gen_mmd   = math.inf
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
        scaler.load_state_dict(ckpt_d["scaler"])
        start_epoch   = ckpt_d["epoch"] + 1
        best_val_loss = ckpt_d.get("best_val_loss", math.inf)
        best_gen_mmd  = ckpt_d.get("best_gen_mmd", math.inf)
        history       = ckpt_d.get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop --------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn = train_one_epoch(
            vae, schedule, style_dn, point_dn,
            trn_loader, opt_s, opt_p, scaler, cfg, epoch, logger, device,
        )
        val = validate(vae, schedule, style_dn, point_dn, val_loader, cfg, device)

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

        if epoch % cfg.vis_every == 0:
            log_generations(
                writer, schedule, style_dn, point_dn, vae, cfg, epoch,
                device, idx_to_name, show_classes,
            )

        gen_metrics = None
        if cfg.eval_gen_metrics and epoch % cfg.eval_every == 0:
            gen_metrics = evaluate_generation_metrics(
                schedule, style_dn, point_dn, vae, cfg, device,
                show_classes, ref_pool, val_ds.dataset,
            )
            if gen_metrics is not None:
                logger.info(
                    f"  gen metrics ({cfg.eval_metric}): "
                    f"MMD={gen_metrics['mmd']:.5f}  "
                    f"COV={gen_metrics['cov']:.3f}  "
                    f"1-NNA={gen_metrics['1nna']:.3f}"
                )
                writer.add_scalar(f"gen_metrics/mmd_{cfg.eval_metric}",   gen_metrics["mmd"],   epoch)
                writer.add_scalar(f"gen_metrics/cov_{cfg.eval_metric}",   gen_metrics["cov"],   epoch)
                writer.add_scalar(f"gen_metrics/1nna_{cfg.eval_metric}",  gen_metrics["1nna"],  epoch)
                for cls_idx, m in gen_metrics["per_class"].items():
                    cls_name = idx_to_name.get(cls_idx, str(cls_idx))
                    writer.add_scalar(f"gen_metrics_per_class/{cls_name}/mmd",  m["mmd"],  epoch)
                    writer.add_scalar(f"gen_metrics_per_class/{cls_name}/cov",  m["cov"],  epoch)
                    writer.add_scalar(f"gen_metrics_per_class/{cls_name}/1nna", m["1nna"], epoch)

        val_loss = val.get("loss_style", math.inf) + val.get("loss_point", math.inf)
        is_best  = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"  New best val loss: {best_val_loss:.5f}")

        is_best_gen = gen_metrics is not None and gen_metrics["mmd"] < best_gen_mmd
        if is_best_gen:
            best_gen_mmd = gen_metrics["mmd"]
            logger.info(f"  New best generation MMD: {best_gen_mmd:.5f}")

        save_state = {
            "epoch":         epoch,
            "style_dn":      style_dn.state_dict(),
            "point_dn":      point_dn.state_dict(),
            "opt_s":         opt_s.state_dict(),
            "opt_p":         opt_p.state_dict(),
            "sched_s":       sched_s.state_dict(),
            "sched_p":       sched_p.state_dict(),
            "scaler":        scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_gen_mmd":  best_gen_mmd,
            "config":        asdict(cfg),
            "history":       history,
        }
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"))
        if is_best:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))
        if is_best_gen:
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best_gen.pt"))

        history_entry = {"epoch": epoch, **{f"trn_{k}": v for k, v in trn.items()},
                                          **{f"val_{k}": v for k, v in val.items()}}
        if gen_metrics is not None:
            history_entry.update({f"gen_{k}": gen_metrics[k] for k in ("mmd", "cov", "1nna")})
        history.append(history_entry)
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

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
