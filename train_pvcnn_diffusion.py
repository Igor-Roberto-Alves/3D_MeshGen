"""
train_pvcnn_diffusion.py
-------------------------
Train the PVCNN2 (PVD-style) raw-point-cloud diffusion model defined in
src/PVCNN2Diffusion.py, conditioned on object category.

Unlike train_diffusion.py (LION two-stage, diffuses a frozen VAE's latent
space), this is a SINGLE denoiser that diffuses the point cloud xyz directly:

  x0 (B, 3, N) real points ──q_sample──► xt (noisy) ──PVCNN2──► ε_pred
                                                            │
                                                     MSE(ε_pred, ε_real)

No VAE involved — this is a standalone alternative pipeline. Reuses
EMA / use_ema / diffusion_loss / set_seed / get_logger / count_parameters
from train_diffusion.py (generic utilities, not modified) so the two
pipelines share the proven training machinery without duplicating it.
"""

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import Ds_point_sampled_already, Ds_point_model
from src.Vae import normalize_pc
from src.Diffusion import CosineSchedule
from src.PVCNN2Diffusion import PVCNN2
from src.metric import generative_metrics
from train_diffusion import (
    EMA, use_ema, diffusion_loss, set_seed, get_logger, count_parameters, log_metrics,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PVCNNDiffusionConfig:
    # --- paths ---
    ckpt_dir:  str = "checkpoints_pvcnn_diff"
    log_dir:   str = "logs_pvcnn_diff"
    data_root: str = "point_clouds"

    # --- data ---
    n_points:    int = 2048
    num_classes: int = 55   # overwritten by whatever classes are actually in data_root

    # --- diffusion schedule ---
    T: int = 1000

    # --- PVCNN2 architecture ---
    embed_dim:                  int   = 64
    use_att:                    bool  = True
    dropout:                    float = 0.1
    width_multiplier:           float = 1.0
    voxel_resolution_multiplier: float = 1.0
    cfg_dropout:                 float = 0.1
    guidance:                    float = 3.0   # CFG scale at inference

    # --- training ---
    epochs:        int   = 300
    batch_size:    int   = 16
    lr:            float = 1e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0
    min_snr_gamma: float = 5.0
    ema_decay:     float = 0.999

    # --- data split ---
    val_split:   float = 0.1
    test_split:  float = 0.5
    num_workers: int   = 4
    pin_memory:  bool  = True

    # --- misc ---
    seed:       int  = 42
    save_every: int  = 10
    log_every:  int  = 50
    device:     str  = "cuda"
    amp:        bool = True
    resume:     int  = 0

    # --- generation visualization ---
    vis_every:     int = 10
    vis_per_class: int = 2

    # --- generation metrics ---
    eval_every:  int = 50
    eval_n_gen:  int = 16
    eval_n_ref:  int = 16
    eval_points: int = 1024
    eval_chunk:  int = 4


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample_class(
    schedule: CosineSchedule,
    model:    PVCNN2,
    cfg:      PVCNNDiffusionConfig,
    cls_idx:  int,
    B:        int,
    device:   torch.device,
) -> torch.Tensor:
    """Full DDPM reverse chain for one class → (B, n_points, 3) xyz."""
    cls    = torch.full((B,), cls_idx, device=device, dtype=torch.long)
    uncond = torch.full((B,), model.uncond_idx, device=device, dtype=torch.long)
    x = schedule.sample(
        model, (3, cfg.n_points),
        condition=cls, uncond=uncond, guidance=cfg.guidance, device=device,
    )  # (B, 3, n_points)
    return x.transpose(1, 2).contiguous()  # (B, n_points, 3)


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
    return (torch.cat(shifted_v).unsqueeze(0), torch.cat(shifted_c).unsqueeze(0))


@torch.no_grad()
def log_generations(
    writer:          SummaryWriter,
    schedule:        CosineSchedule,
    model:           PVCNN2,
    cfg:             PVCNNDiffusionConfig,
    epoch:           int,
    device:          torch.device,
    class_names:     dict,
    present_classes: list,
):
    model.eval()
    for cls_idx in present_classes[:4]:
        B = cfg.vis_per_class
        xyz_out = sample_class(schedule, model, cfg, cls_idx, B, device).cpu()
        for i in range(B):
            gen_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(cfg.n_points, -1)
            verts, clrs = _side_by_side([xyz_out[i]], [gen_c])
            cls_name = class_names.get(cls_idx, str(cls_idx))
            writer.add_mesh(f"generated/{cls_name}/sample{i}", vertices=verts, colors=clrs, global_step=epoch)
    model.train()


@torch.no_grad()
def evaluate_generation_metrics(
    writer:       SummaryWriter,
    schedule:     CosineSchedule,
    model:        PVCNN2,
    cfg:          PVCNNDiffusionConfig,
    epoch:        int,
    device:       torch.device,
    class_names:  dict,
    ref_by_class: dict,
    logger:       logging.Logger,
) -> dict:
    model.eval()
    agg = {"mmd": [], "cov": [], "nna_1nn": []}

    for cls_idx, ref_xyz in ref_by_class.items():
        n_ref = min(cfg.eval_n_ref, ref_xyz.shape[0])
        if n_ref < 2:
            continue
        ref_idx = torch.randperm(ref_xyz.shape[0], device=device)[:n_ref]
        ref = ref_xyz[ref_idx]

        gen = sample_class(schedule, model, cfg, cls_idx, cfg.eval_n_gen, device)

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

    model.train()
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

def _batch_xyz(points: torch.Tensor, device: torch.device) -> torch.Tensor:
    """points: (B, N, 6) xyz+normals -> normalized (B, 3, N) xyz, channel-first."""
    points = points.to(device, non_blocking=True)
    xyz = normalize_pc(points)[..., :3]     # (B, N, 3)
    return xyz.transpose(1, 2).contiguous()  # (B, 3, N)


def train_one_epoch(
    schedule: CosineSchedule,
    model:    PVCNN2,
    loader:   DataLoader,
    opt:      torch.optim.Optimizer,
    scaler:   torch.amp.GradScaler,
    cfg:      PVCNNDiffusionConfig,
    epoch:    int,
    logger:   logging.Logger,
    device:   torch.device,
    ema:      "EMA | None" = None,
) -> dict:
    model.train()
    totals, n_batches = {}, 0
    gamma = cfg.min_snr_gamma if cfg.min_snr_gamma > 0 else None

    for batch_idx, (points, class_idx) in enumerate(loader):
        x0 = _batch_xyz(points, device)
        class_idx = class_idx.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss = diffusion_loss(schedule, model, x0, class_idx, gamma)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)

        totals["loss"] = totals.get("loss", 0.0) + loss.item()
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
    schedule: CosineSchedule,
    model:    PVCNN2,
    loader:   DataLoader,
    cfg:      PVCNNDiffusionConfig,
    device:   torch.device,
) -> dict:
    model.eval()
    totals, n_batches = {}, 0
    gamma = cfg.min_snr_gamma if cfg.min_snr_gamma > 0 else None

    for points, class_idx in loader:
        x0 = _batch_xyz(points, device)
        class_idx = class_idx.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            loss = diffusion_loss(schedule, model, x0, class_idx, gamma)
        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ============================================================
# Main
# ============================================================

def main(cfg: PVCNNDiffusionConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    # ---- Data -------------------------------------------------
    base_ds = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    name_map = Ds_point_model.map()
    idx_to_name = {idx: name_map.get(cid, cid) for cid, idx in base_ds.class_to_idx.items()}
    present_classes = sorted(idx_to_name.keys())

    n_classes_actual = len(base_ds.class_to_idx)
    if n_classes_actual != cfg.num_classes:
        logger.info(
            f"num_classes={cfg.num_classes} in config but {n_classes_actual} class(es) "
            f"found in '{cfg.data_root}' ({[idx_to_name[i] for i in present_classes]}) "
            f"— using {n_classes_actual}."
        )
        cfg.num_classes = n_classes_actual

    indices = torch.randperm(len(base_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n = max(1, int(len(base_ds) * cfg.val_split))
    train_idx = indices[val_n:]
    val_pool  = indices[:val_n]

    if cfg.test_split > 0:
        test_n = min(max(1, int(len(val_pool) * cfg.test_split)), len(val_pool) - 1)
        test_idx = val_pool[:test_n]
        val_idx  = val_pool[test_n:]
    else:
        test_idx, val_idx = [], val_pool

    trn_ds  = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=True), train_idx)
    val_ds  = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)
    test_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=False), test_idx)
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    trn_loader  = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader  = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = (DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
                   if test_idx else None)

    def _build_ref_cache(subset) -> dict:
        cache: dict = {}
        for feats, cls in subset:
            cache.setdefault(int(cls.item()), []).append(feats[:, :3])
        for cls_idx, pts_list in cache.items():
            xyz = torch.stack(pts_list).to(device)
            cache[cls_idx] = normalize_pc(xyz)
        return cache

    logger.info("Building per-class reference caches (val + test)...")
    ref_by_class      = _build_ref_cache(val_ds)
    test_ref_by_class = _build_ref_cache(test_ds)

    # ---- Model --------------------------------------------------
    schedule = CosineSchedule(T=cfg.T).to(device)
    model = PVCNN2(
        num_categories=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        use_att=cfg.use_att,
        dropout=cfg.dropout,
        extra_feature_channels=0,
        width_multiplier=cfg.width_multiplier,
        voxel_resolution_multiplier=cfg.voxel_resolution_multiplier,
        cfg_dropout=cfg.cfg_dropout,
    ).to(device)
    logger.info(f"PVCNN2 params: {count_parameters(model):,}")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])

    amp_on = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    ema_on = cfg.ema_decay > 0
    ema = EMA(model, cfg.ema_decay) if ema_on else None
    if ema_on:
        logger.info(f"EMA ativado (decay={cfg.ema_decay}, warmup adaptativo)")

    # ---- Resume ---------------------------------------------------
    start_epoch, best_val_loss, history = 0, math.inf, []
    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", math.inf)
        history = ckpt.get("history", [])
        if ema_on and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], device)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop --------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn = train_one_epoch(schedule, model, trn_loader, opt, scaler, cfg, epoch, logger, device, ema)
        val = validate(schedule, model, val_loader, cfg, device)

        sched.step()
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  lr={sched.get_last_lr()[0]:.2e}  time={elapsed:.1f}s\n"
            f"  trn: {trn}\n  val: {val}"
        )
        writer.add_scalar("train/lr", sched.get_last_lr()[0], epoch)
        log_metrics(writer, trn, "train", epoch)
        log_metrics(writer, val, "val", epoch)

        ema_pairs = [(ema, model)] if ema_on else []

        if epoch % cfg.vis_every == 0:
            with use_ema(ema_pairs):
                log_generations(writer, schedule, model, cfg, epoch, device, idx_to_name, present_classes)

        eval_summary = {}
        if epoch % cfg.eval_every == 0:
            with use_ema(ema_pairs):
                eval_summary = evaluate_generation_metrics(
                    writer, schedule, model, cfg, epoch, device, idx_to_name, ref_by_class, logger
                )

        val_loss = val.get("loss", math.inf)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"  New best val loss: {best_val_loss:.5f}")

        save_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema_on else None,
            "best_val_loss": best_val_loss,
            "config": asdict(cfg),
            "history": history,
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

    # ---- Final test evaluation ----------------------------------------
    if test_ref_by_class:
        logger.info("=== Final test-set evaluation ===")
        test_val = validate(schedule, model, test_loader, cfg, device)
        log_metrics(writer, test_val, "test", cfg.epochs)
        logger.info(f"  test losses: {test_val}")

        with use_ema([(ema, model)] if ema_on else []):
            test_gen = evaluate_generation_metrics(
                writer, schedule, model, cfg, cfg.epochs, device, idx_to_name, test_ref_by_class, logger
            )
        with open(os.path.join(cfg.log_dir, "test_metrics.json"), "w") as f:
            json.dump({"losses": test_val, "generation": test_gen}, f, indent=2)
        logger.info(f"  test metrics saved -> {cfg.log_dir}/test_metrics.json")

    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> PVCNNDiffusionConfig:
    cfg = PVCNNDiffusionConfig()
    p = argparse.ArgumentParser(description="Train PVCNN2 (PVD-style) raw point-cloud diffusion")
    for name, val in asdict(cfg).items():
        t = type(val)
        if t is bool:
            p.add_argument(f"--{name}", default=val, type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{name}", default=val, type=t)
    return PVCNNDiffusionConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
