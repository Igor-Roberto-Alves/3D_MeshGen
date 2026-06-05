"""
train_vae.py
------------
Training loop for the PVCNN-inspired Point-Cloud VAE.

Usage
-----
python train_vae.py [--config config.yaml]   # YAML overrides (optional)
python train_vae.py --epochs 200 --latent_dim 512 --beta 1.0

Dataset expected: Ds_point_sampled_already at root "point_clouds/"
(produced by dataset.py).
"""
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import math
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

from src.dataset import Ds_point_sampled_already
from src.Encoder import Encoder
from src.Decoder import Decoder
from src.metric import vae_loss, chamfer_distance_knn, f_score, normal_consistency


# ============================================================
# Configuration
# ============================================================

# ============================================================
# TensorBoard helpers
# ============================================================

def log_metrics_tensorboard(
    writer: SummaryWriter,
    metrics: dict[str, float],
    prefix: str,
    epoch: int,
):
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
    """
    Logs reconstructed point clouds to TensorBoard.

    Creates 3D meshes:
        GT
        Reconstruction
        Sampled prior

    TensorBoard mesh format:
        vertices -> (B, N, 3)
        colors   -> (B, N, 3)
    """

    model.eval()

    points = next(iter(loader))
    points = points.to(device)

    points = normalise_batch(points)

    target_xyz = points[..., :3]

    coarse, refined, normals, mu, logvar = model(points)

    refined = refined.detach().float().cpu()
    target_xyz = target_xyz.detach().float().cpu()

    B = min(max_items, refined.shape[0])

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    gt_colors = torch.zeros_like(target_xyz[:B])
    gt_colors[..., 1] = 255

    pred_colors = torch.zeros_like(refined[:B])
    pred_colors[..., 0] = 255

    # --------------------------------------------------------
    # GT meshes
    # --------------------------------------------------------

    writer.add_mesh(
        tag="reconstruction/ground_truth",
        vertices=target_xyz[:B],
        colors=gt_colors,
        global_step=epoch,
    )

    # --------------------------------------------------------
    # Reconstructions
    # --------------------------------------------------------

    writer.add_mesh(
        tag="reconstruction/prediction",
        vertices=refined[:B],
        colors=pred_colors,
        global_step=epoch,
    )

    # --------------------------------------------------------
    # Prior samples
    # --------------------------------------------------------

    samples = model.sample(B, device).detach().float().cpu()

    sample_colors = torch.zeros_like(samples)
    sample_colors[..., 2] = 255

    writer.add_mesh(
        tag="samples/prior",
        vertices=samples,
        colors=sample_colors,
        global_step=epoch,
    )

    model.train()

@dataclass
class TrainConfig:
    # --- paths ---
    data_root:     str  = "point_clouds"
    ckpt_dir:      str  = "checkpoints"
    log_dir:       str  = "logs"

    # --- architecture ---
    latent_dim:    int   = 256
    num_points:    int   = 2048
    decoder_hidden: int  = 512

    # --- training ---
    epochs:        int   = 200
    batch_size:    int   = 32
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0

    # --- VAE ---
    beta_start:    float = 0.0    # β annealing: start value
    beta_end:      float = 1.0    # β annealing: final value
    beta_epochs:   int   = 100    # epochs to reach beta_end
    recon_loss:    str   = "chamfer"   # "chamfer" | "emd" | "both"
    emd_weight:    float = 0.5    # used only when recon_loss="both"

    # --- data ---
    val_split:     float = 0.1
    num_workers:   int   = 4
    pin_memory:    bool  = True

    # --- misc ---
    seed:          int   = 42
    save_every:    int   = 10    # save checkpoint every N epochs
    log_every:     int   = 50    # log metrics every N batches
    device:        str   = "cuda"
    amp:           bool  = True   # automatic mixed precision
    resume:        int   = 0     # epoch to resume from (default: 0, i.e. no resume)


# ============================================================
# Full VAE model wrapper
# ============================================================

class PointCloudVAE(nn.Module):
    """Thin wrapper that glues Encoder + Decoder."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.encoder = Encoder(latent_dim=cfg.latent_dim, in_channels=6)
        self.decoder = Decoder(
            latent_dim=cfg.latent_dim,
            num_points=cfg.num_points,
            hidden=cfg.decoder_hidden,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·exp(0.5·logvar),  ε ~ N(0, I)."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (B, N, 6)

        Returns
        -------
        coarse, refined, normals : decoder outputs
        mu, logvar               : encoder outputs
        """
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        coarse, refined, normals = self.decoder(z)
        return coarse, refined, normals, mu, logvar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n point clouds from the prior p(z) = N(0, I)."""
        z = torch.randn(n, self.encoder.latent_dim, device=device)
        _, refined, _ = self.decoder(z)
        return refined


# ============================================================
# Utilities
# ============================================================

def beta_schedule(epoch: int, cfg: TrainConfig) -> float:
    """Linear β annealing from beta_start → beta_end over beta_epochs."""
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


def normalise_batch(points: torch.Tensor) -> torch.Tensor:
    """
    Centre and scale each point cloud independently.
    points : (B, N, 6)  [xyz | normals]
    """
    xyz    = points[..., :3]
    centre = xyz.mean(dim=1, keepdim=True)
    xyz    = xyz - centre
    scale  = xyz.norm(dim=2, keepdim=True).max(dim=1, keepdim=True).values.clamp(min=1e-6)
    xyz    = xyz / scale
    return torch.cat([xyz, points[..., 3:]], dim=2)


# ============================================================
# Collate – pads to cfg.num_points
# ============================================================

def make_collate(num_points: int):
    def collate(batch):
        features_list = []
        for _, feat in batch:                             # feat: (N', 6)
            N = feat.shape[0]
            if N >= num_points:
                idx  = torch.randperm(N)[:num_points]
                feat = feat[idx]
            else:
                # random repeat-pad
                pad  = num_points - N
                idx  = torch.randint(0, N, (pad,))
                feat = torch.cat([feat, feat[idx]], dim=0)
            features_list.append(feat)
        return torch.stack(features_list, dim=0)          # (B, num_points, 6)
    return collate


# ============================================================
# Train / Val step
# ============================================================

def train_one_epoch(
    model:     PointCloudVAE,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    scaler:    torch.cuda.amp.GradScaler,
    cfg:       TrainConfig,
    epoch:     int,
    logger:    logging.Logger,
    device:    torch.device,
) -> dict[str, float]:

    model.train()
    beta = beta_schedule(epoch, cfg)

    totals: dict[str, float] = {}
    n_batches = 0

    for batch_idx, points in enumerate(loader):
        points = points.to(device, non_blocking=True)   # (B, N, 6)
        points = normalise_batch(points)
        target_xyz = points[..., :3]                    # (B, N, 3)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            coarse, refined, normals, mu, logvar = model(points)

            losses = vae_loss(
                pred_xyz   = refined,
                target_xyz = target_xyz,
                mu         = mu,
                logvar     = logvar,
                beta       = beta,
                recon_loss = cfg.recon_loss,
                emd_weight = cfg.emd_weight,
            )

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimiser)
        scaler.update()

        # accumulate
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            logger.info(
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  "
                f"β={beta:.3f}  "
                + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model:  PointCloudVAE,
    loader: DataLoader,
    cfg:    TrainConfig,
    epoch:  int,
    device: torch.device,
) -> dict[str, float]:

    model.eval()
    beta = beta_schedule(epoch, cfg)

    totals: dict[str, float] = {}
    n_batches = 0

    for points in loader:
        points = points.to(device, non_blocking=True)
        points = normalise_batch(points)
        target_xyz = points[..., :3]
        target_nrm = points[..., 3:]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            coarse, refined, normals, mu, logvar = model(points)

            losses = vae_loss(
                pred_xyz   = refined,
                target_xyz = target_xyz,
                mu         = mu,
                logvar     = logvar,
                beta       = beta,
                recon_loss = cfg.recon_loss,
                emd_weight = cfg.emd_weight,
            )

        # extra eval metrics (no grad needed)
        fs  = f_score(refined, target_xyz, threshold=0.01)
        nc  = normal_consistency(refined, normals, target_xyz, target_nrm)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["f_score"]  = totals.get("f_score",  0.0) + fs["f_score"].item()
        totals["normal_c"] = totals.get("normal_c", 0.0) + nc.item()
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
    

    device = torch.device(
        cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ---- dataset -------------------------------------------------------
    full_ds = Ds_point_sampled_already(root=cfg.data_root)
    val_n   = max(1, int(len(full_ds) * cfg.val_split))
    trn_n   = len(full_ds) - val_n
    trn_ds, val_ds = random_split(
        full_ds, [trn_n, val_n],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    collate = make_collate(cfg.num_points)
    trn_loader = DataLoader(
        trn_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate,
    )
    logger.info(f"Dataset: {trn_n} train / {val_n} val samples")

    # ---- model ---------------------------------------------------------
    model = PointCloudVAE(cfg).to(device)
    logger.info(f"Parameters: {count_parameters(model):,}")

    # ---- optimiser / scheduler ----------------------------------------
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup = LinearLR(
        optimiser,
        start_factor=1e-3, end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimiser,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.lr * 1e-2,
    )
    scheduler = SequentialLR(
        optimiser, schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs]
    )

    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=cfg.amp and device.type == "cuda"
    )

    # ---- resume --------------------------------------------------------
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

    # ---- training loop -------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        trn_metrics = train_one_epoch(
            model, trn_loader, optimiser, scaler, cfg, epoch, logger, device
        )
        val_metrics = validate(model, val_loader, cfg, epoch, device)

        scheduler.step()
        elapsed = time.time() - t0

        # log epoch summary
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
        # ========================================================
        # TensorBoard logging
        # ========================================================

        writer.add_scalar(
            "train/lr",
            scheduler.get_last_lr()[0],
            epoch
        )

        writer.add_scalar(
            "train/beta",
            beta_schedule(epoch, cfg),
            epoch
        )

        log_metrics_tensorboard(writer, trn_metrics, "train", epoch)
        log_metrics_tensorboard(writer, val_metrics, "val", epoch)

        # latent stats
        writer.add_scalar(
            "latent/best_val_cd",
            best_val_cd,
            epoch
        )

        # reconstruction visualisation
        if epoch % 5 == 0:
            log_reconstructions(
                writer,
                model,
                val_loader,
                device,
                epoch,
                max_items=4,
            )

                # ---- checkpointing -------------------------------------------
        record = {"epoch": epoch, **{f"trn_{k}": v for k, v in trn_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

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

        # always save latest
        torch.save(save_state, os.path.join(cfg.ckpt_dir, "latest.pt"))

        # periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(
                save_state,
                os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt")
            )

        # best model (by val Chamfer distance)
        val_cd_key = "val_cd" if "val_cd" in val_metrics else "val_recon"
        val_cd = val_metrics.get(val_cd_key, math.inf)
        if val_cd < best_val_cd:
            best_val_cd = val_cd
            save_state["best_val_cd"] = best_val_cd
            torch.save(save_state, os.path.join(cfg.ckpt_dir, "best.pt"))
            logger.info(f"  ✓ New best val CD: {best_val_cd:.6f}")

        # save history JSON
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    writer.close()
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p   = argparse.ArgumentParser(description="Train PVCNN Point-Cloud VAE")
    for field_name, field_val in asdict(cfg).items():
        t = type(field_val)
        if t is bool:
            p.add_argument(f"--{field_name}", default=field_val,
                           type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{field_name}", default=field_val, type=t)
    

    args = vars(p.parse_args())
    return TrainConfig(**args)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)