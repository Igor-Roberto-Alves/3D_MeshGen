"""
train_vae_flat.py
------------------
Training loop para o VaeFlat: latente UNICO e flat, sem split
global/local, sem "pontos latentes". Decoder StyleGAN-like (canvas
constante + AdaGN). Ver src/vae_flat.py e src/decoder_flat.py.

Diferencas em relacao a train_vae_up.py:
  - Sem coarse_weight (removido tambem do VaeUp — loss sem pressao de KL
    que so incentivava memorizacao).
  - Sem zg_dropout_p (nao ha mais z_g separado de z_l pra "vazar" info).
  - KL unica (kl_divergence 2-D: piso de free-bits por canal, ja
    implementado em src/metric.py).
  - beta_hold_epochs: mantem beta=0 fixo por N epocas antes do ramp
    comecar, pra dar tempo do decoder aprender a depender de z antes da
    KL apertar (mitiga o colapso que aparecia ja na epoca 2 no VaeUp).
"""

import argparse
import json
import logging
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
from src.metric import chamfer_distance_knn, emd_approx, kl_divergence, f_score
from src.vae_flat import VaeFlat
from src.Vae import normalize_pc


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainConfig:
    # --- paths ---
    data_root:        str   = "point_clouds"
    ckpt_dir:         str   = "checkpoints_flat"
    log_dir:          str   = "logs_flat"

    # --- architecture ---
    latent_dim:       int   = 512    # dimensao do vetor latente unico
    n_latent:         int   = 512    # ancoras do canvas constante (antes do folding)
    n_points:         int   = 2048   # pontos de saida (deve bater com o dataset)
    seed_dim:         int   = 128    # largura do embedding por ancora do canvas
    in_channels:      int   = 6

    # --- capacidade do decoder (escale so por flag, sem editar codigo) ---
    decoder_hidden_dim:  int = 384   # largura inicial dos estagios PVConv
    decoder_stages:      int = 4     # profundidade (numero de estagios PVConv)
    decoder_fold_dim:    int = 64    # largura da feature que entra no fold_mlp
    decoder_fold_hidden: int = 256   # largura interna do fold_mlp
    decoder_resolution:  int = 16    # grade de voxelizacao de cada PVConv (custo ~quadratico nos canais)

    # --- capacidade do encoder (escale so por flag, sem editar codigo) ---
    encoder_hidden_dim:  int = 64    # largura do 1o estagio PVConv
    encoder_out_dim:     int = 256   # largura do ultimo estagio — gargalo real antes do max-pool/fc_mu/fc_logvar
    encoder_stages:      int = 3     # profundidade (numero de estagios PVConv)
    encoder_resolution:  int = 32    # grade de voxelizacao do 1o estagio (dobra a cada estagio seguinte)

    # --- training ---
    epochs:           int   = 200
    batch_size:       int   = 8
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    warmup_epochs:    int   = 10
    grad_clip:        float = 1.0

    # --- VAE beta scheduling (KL annealing) ---
    beta_start:       float = 0.0
    beta_end:         float = 1.0
    beta_hold_epochs: int   = 10    # beta=0 fixo por essas epocas antes do ramp comecar
    beta_epochs:      int   = 60    # duracao do ramp APOS o hold

    # --- KL ---
    free_bits:        float = 0.03  # piso por canal do latente flat (ver metric.kl_divergence)

    # --- reconstruction loss ---
    recon_loss:       str   = "both"
    emd_weight:       float = 0.5
    emd_iters:        int   = 15
    emd_n_subsample:  int   = 512

    # --- data ---
    val_split:        float = 0.1
    num_workers:      int   = 4
    pin_memory:       bool  = True

    # --- misc ---
    seed:             int   = 42
    save_every:       int   = 5
    log_every:        int   = 50
    device:           str   = "cuda"
    amp:              bool  = True
    resume:           int   = 0
    compile_model:    bool  = False
    grad_hist_every:  int   = 20


# ============================================================
# TensorBoard helpers
# ============================================================

def log_metrics_tensorboard(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


def _side_by_side(clouds, colors, gap=2.5):
    shifted_v, shifted_c = [], []
    x_cursor = 0.0
    for v, c in zip(clouds, colors):
        v = v.clone()
        v[:, 0] -= v[:, 0].mean()
        v[:, 0] += x_cursor
        half_width = (v[:, 0].max() - v[:, 0].min()).item() * 0.5
        x_cursor  += half_width * 2 + gap
        shifted_v.append(v)
        shifted_c.append(c)
    verts = torch.cat(shifted_v, dim=0).unsqueeze(0)
    clrs  = torch.cat(shifted_c, dim=0).unsqueeze(0)
    return verts, clrs


def log_grad_norms(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"grad_norm/{name}", param.grad.norm().item(), epoch)


def log_grad_histograms(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)


@torch.no_grad()
def log_reconstructions(writer, model, loader, device, epoch, split="train", max_items=4):
    model.eval()
    points, _ = next(iter(loader))
    points = points.to(device)
    N = points.shape[1]

    xyz_out, *_ = model(points)
    gt    = normalize_pc(points)[..., :3].detach().float().cpu()
    recon = xyz_out.detach().float().cpu()
    B     = min(max_items, gt.shape[0])

    prior_samples = model.generate(num_samples=B, num_points=N, device=device)
    prior_samples = prior_samples.detach().float().cpu()

    for i in range(B):
        gt_v  = gt[i];    gt_c  = torch.tensor([[0, 220, 0]],   dtype=torch.uint8).expand(N, -1)
        rec_v = recon[i]; rec_c = torch.tensor([[220, 0, 0]],   dtype=torch.uint8).expand(N, -1)
        pri_v = prior_samples[i]
        pri_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(N, -1)

        verts, clrs = _side_by_side([gt_v, rec_v, pri_v], [gt_c, rec_c, pri_c])
        writer.add_mesh(f"recon_{split}/sample_{i}", vertices=verts, colors=clrs, global_step=epoch)

    model.train()


# ============================================================
# Loss (latente unico — nao usa src.metric.vae_loss, que assume 2 KLs)
# ============================================================

def flat_vae_loss(pred_xyz, target_xyz, mu, logvar, beta, free_bits,
                   recon_loss="both", emd_weight=0.5, emd_iters=15, emd_n_subsample=0):
    if recon_loss == "chamfer":
        recon, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        out = {"recon": recon, "cd_forward": cd_f, "cd_backward": cd_b}
    elif recon_loss == "emd":
        recon = emd_approx(pred_xyz, target_xyz, n_iters=emd_iters, n_subsample=emd_n_subsample)
        out = {"recon": recon}
    elif recon_loss == "both":
        cd, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        emd = emd_approx(pred_xyz, target_xyz, n_iters=emd_iters, n_subsample=emd_n_subsample)
        recon = (1 - emd_weight) * cd + emd_weight * emd
        out = {"recon": recon, "cd": cd, "emd": emd, "cd_forward": cd_f, "cd_backward": cd_b}
    else:
        raise ValueError(f"Unknown recon_loss: '{recon_loss}'.")

    kl = kl_divergence(mu, logvar, free_bits=free_bits)
    out["kl"] = kl
    out["total"] = recon + beta * kl
    return out


# ============================================================
# Utilities
# ============================================================

def beta_schedule(epoch, cfg):
    if epoch < cfg.beta_hold_epochs:
        return cfg.beta_start
    t_epoch = epoch - cfg.beta_hold_epochs
    if t_epoch >= cfg.beta_epochs:
        return cfg.beta_end
    t = t_epoch / cfg.beta_epochs
    return cfg.beta_start + t * (cfg.beta_end - cfg.beta_start)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_logger(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vae_flat_train")
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
    beta      = beta_schedule(epoch, cfg)
    totals    = {}
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        points, _ = data
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, xyz_coarse, mu, logvar = model(points)
            losses = flat_vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu=mu,
                logvar=logvar,
                beta=beta,
                free_bits=cfg.free_bits,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                emd_iters=cfg.emd_iters,
                emd_n_subsample=cfg.emd_n_subsample,
            )

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
                f"  Epoch {epoch:03d}  Batch {batch_idx+1}/{len(loader)}  beta={beta:.4f}  "
                + "  ".join(f"{k}={v:.5f}" for k, v in avg.items())
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, cfg, epoch, device):
    """Retorna (metrics, kl_per_channel, latent_stats).
    kl_per_channel e (latent_dim,) — KL medio por canal sobre todo o split de
    validacao, usado pro log de 'canais ativos' (ver log_latent_activity).
    latent_stats traz o detector de collapse independente do piso de
    free-bits: std de mu POR CANAL atraves das amostras do val (se mu nao
    varia entre shapes diferentes, o codigo nao carrega informacao) e o
    sigma medio do posterior."""
    model.eval()
    beta      = beta_schedule(epoch, cfg)
    totals    = {}
    n_batches = 0
    kl_sum    = None
    n_samples = 0
    mu_sum = mu_sq_sum = sigma_sum = None

    for points, _ in loader:
        points     = points.to(device, non_blocking=True)
        target_xyz = normalize_pc(points)[..., :3]

        with torch.autocast(device_type=device.type, enabled=cfg.amp):
            xyz_out, xyz_coarse, mu, logvar = model(points)
            losses = flat_vae_loss(
                pred_xyz=xyz_out,
                target_xyz=target_xyz,
                mu=mu,
                logvar=logvar,
                beta=beta,
                free_bits=cfg.free_bits,
                recon_loss=cfg.recon_loss,
                emd_weight=cfg.emd_weight,
                emd_iters=cfg.emd_iters,
                emd_n_subsample=cfg.emd_n_subsample,
            )

        fs05 = f_score(xyz_out, target_xyz, threshold=0.05)
        fs10 = f_score(xyz_out, target_xyz, threshold=0.10)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        totals["f05"] = totals.get("f05", 0.0) + fs05["f_score"].item()
        totals["f10"] = totals.get("f10", 0.0) + fs10["f_score"].item()
        n_batches += 1

        mu_c, logvar_c = mu.float().clamp(-10.0, 10.0), logvar.float().clamp(-10.0, 10.0)
        kl_elem = -0.5 * (1.0 + logvar_c - mu_c.pow(2) - logvar_c.exp())  # (B, D)
        batch_sum = kl_elem.sum(dim=0)
        kl_sum    = batch_sum if kl_sum is None else kl_sum + batch_sum
        n_samples += points.shape[0]

        sigma_b = (0.5 * logvar_c).exp().sum(dim=0)
        if mu_sum is None:
            mu_sum, mu_sq_sum, sigma_sum = mu_c.sum(dim=0), mu_c.pow(2).sum(dim=0), sigma_b
        else:
            mu_sum    += mu_c.sum(dim=0)
            mu_sq_sum += mu_c.pow(2).sum(dim=0)
            sigma_sum += sigma_b

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    n = max(n_samples, 1)
    kl_per_channel = (kl_sum / n).cpu()
    mu_mean        = mu_sum / n
    mu_std_per_ch  = (mu_sq_sum / n - mu_mean.pow(2)).clamp(min=0.0).sqrt().cpu()  # (D,)
    latent_stats = {
        "mu_std_per_channel": mu_std_per_ch,
        "sigma_mean":         (sigma_sum / n).mean().item(),
    }
    return metrics, kl_per_channel, latent_stats


def log_latent_activity(writer, kl_per_channel, free_bits, epoch, log_hist):
    """Canal 'ativo' = KL medio (sobre o split de val) estritamente acima do
    piso de free-bits — canais parados exatamente no piso nao contam como
    ativos (estao la so por causa do clamp, nao porque carregam informacao)."""
    threshold = free_bits + 1e-3
    active = (kl_per_channel > threshold).sum().item()
    total  = kl_per_channel.numel()
    writer.add_scalar("latent/active_units", active, epoch)
    writer.add_scalar("latent/active_units_frac", active / total, epoch)
    writer.add_scalar("latent/kl_per_channel_mean", kl_per_channel.mean().item(), epoch)
    writer.add_scalar("latent/kl_per_channel_max",  kl_per_channel.max().item(), epoch)
    if log_hist:
        writer.add_histogram("latent/kl_per_channel", kl_per_channel, epoch)
    return active, total


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

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True

    # ---- Datasets -------------------------------------------------------
    base_ds  = Ds_point_sampled_already(root=cfg.data_root, augment=False)
    indices  = torch.randperm(len(base_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_n    = max(1, int(len(base_ds) * cfg.val_split))
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    trn_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=True),  train_idx)
    val_ds = torch.utils.data.Subset(Ds_point_sampled_already(root=cfg.data_root, augment=False), val_idx)

    _dl_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    trn_loader = DataLoader(trn_ds, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=True, **_dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, **_dl_kwargs)
    logger.info(f"Dataset: {len(train_idx)} train / {len(val_idx)} val samples")

    # ---- Model ----------------------------------------------------------
    model = VaeFlat(
        latent_dim=cfg.latent_dim,
        in_channels=cfg.in_channels,
        n_latent=cfg.n_latent,
        n_points=cfg.n_points,
        seed_dim=cfg.seed_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        n_stages=cfg.decoder_stages,
        fold_dim=cfg.decoder_fold_dim,
        fold_hidden=cfg.decoder_fold_hidden,
        resolution=cfg.decoder_resolution,
        encoder_hidden_dim=cfg.encoder_hidden_dim,
        encoder_out_dim=cfg.encoder_out_dim,
        encoder_stages=cfg.encoder_stages,
        encoder_resolution=cfg.encoder_resolution,
    ).to(device)
    logger.info(
        f"VaeFlat  |  params: {count_parameters(model):,}  "
        f"(encoder: {count_parameters(model.encoder):,}  decoder: {count_parameters(model.decoder):,})"
    )
    logger.info(
        f"  latent_dim={cfg.latent_dim}  n_latent={cfg.n_latent}  "
        f"ratio={cfg.n_points // cfg.n_latent}  seed_dim={cfg.seed_dim}"
    )
    logger.info(
        f"  decoder_hidden_dim={cfg.decoder_hidden_dim}  decoder_stages={cfg.decoder_stages}  "
        f"decoder_fold_dim={cfg.decoder_fold_dim}  decoder_fold_hidden={cfg.decoder_fold_hidden}  "
        f"decoder_resolution={cfg.decoder_resolution}"
    )
    logger.info(
        f"  encoder_hidden_dim={cfg.encoder_hidden_dim}  encoder_out_dim={cfg.encoder_out_dim}  "
        f"encoder_stages={cfg.encoder_stages}  encoder_resolution={cfg.encoder_resolution}"
    )
    logger.info(
        f"  recon_loss={cfg.recon_loss}  emd_iters={cfg.emd_iters}  "
        f"emd_n_subsample={cfg.emd_n_subsample}  "
        f"beta_hold_epochs={cfg.beta_hold_epochs}  beta_epochs={cfg.beta_epochs}  "
        f"free_bits={cfg.free_bits}"
    )

    if cfg.compile_model and hasattr(torch, "compile"):
        logger.info("Compilando modelo com torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # ---- Optimiser & scheduler ------------------------------------------
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup    = LinearLR(optimiser, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine    = CosineAnnealingLR(optimiser, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 1e-2)
    scheduler = SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # ---- Resume ---------------------------------------------------------
    start_epoch = 0
    best_val_f10 = 0.0
    history: list = []

    resume_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if os.path.exists(resume_path) and cfg.resume:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_f10 = ckpt.get("best_val_f10", 0.0)
        history      = ckpt.get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop --------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        t0          = time.time()
        trn_metrics = train_one_epoch(model, trn_loader, optimiser, scaler, cfg, epoch, logger, device)
        val_metrics, kl_per_channel, latent_stats = validate(model, val_loader, cfg, epoch, device)
        scheduler.step()

        elapsed = time.time() - t0
        trn_str = "  ".join(f"trn_{k}={v:.5f}" for k, v in trn_metrics.items())
        val_str = "  ".join(f"val_{k}={v:.5f}" for k, v in val_metrics.items())
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  beta={beta_schedule(epoch, cfg):.4f}  "
            f"time={elapsed:.1f}s\n  {trn_str}\n  {val_str}"
        )

        writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train/beta", beta_schedule(epoch, cfg),  epoch)
        log_metrics_tensorboard(writer, trn_metrics, "train", epoch)
        log_metrics_tensorboard(writer, val_metrics, "val",   epoch)

        active, total = log_latent_activity(
            writer, kl_per_channel, cfg.free_bits, epoch,
            log_hist=(epoch % cfg.grad_hist_every == 0),
        )
        # Detector de collapse independente do piso de free-bits (AU de
        # Burda et al.): canal 'AU' = std de mu atraves do val set > 0.1.
        mu_std = latent_stats["mu_std_per_channel"]
        au = (mu_std > 0.1).sum().item()
        writer.add_scalar("latent/au_mu",       au,                        epoch)
        writer.add_scalar("latent/mu_std_mean", mu_std.mean().item(),      epoch)
        writer.add_scalar("latent/mu_std_max",  mu_std.max().item(),       epoch)
        writer.add_scalar("latent/sigma_mean",  latent_stats["sigma_mean"], epoch)
        logger.info(
            f"  canais ativos: {active}/{total}  ({100*active/total:.1f}%)  |  "
            f"AU(mu-std>0.1): {au}/{total}  mu_std={mu_std.mean().item():.3f}  "
            f"sigma={latent_stats['sigma_mean']:.3f}"
        )

        if epoch >= cfg.beta_hold_epochs + cfg.beta_epochs:
            log_metrics_tensorboard(writer, trn_metrics, "post_beta/train", epoch)
            log_metrics_tensorboard(writer, val_metrics, "post_beta/val",   epoch)

        if epoch % 5 == 0:
            log_reconstructions(writer, model, trn_loader, device, epoch, split="train", max_items=4)
            log_reconstructions(writer, model, val_loader,  device, epoch, split="val",   max_items=4)
            log_grad_norms(writer, model, epoch)

        if epoch % cfg.grad_hist_every == 0:
            log_grad_histograms(writer, model, epoch)

        val_f10 = val_metrics.get("f10", 0.0)
        is_best = (epoch >= cfg.beta_hold_epochs + cfg.beta_epochs) and (val_f10 > best_val_f10)
        if is_best:
            best_val_f10 = val_f10
            logger.info(f"  New best val F@0.10: {best_val_f10:.4f}")

        save_state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimiser":    optimiser.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "best_val_f10": best_val_f10,
            "config":       asdict(cfg),
            "history":      history,
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
    p   = argparse.ArgumentParser(description="Train Flat-Latent Point-Cloud VAE (constant canvas + AdaGN)")
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