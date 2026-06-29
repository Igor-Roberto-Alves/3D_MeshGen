"""Training script for SetVAE on point clouds."""
import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.SetVAE import SetVAE, normalize_pc
from src.dataset import Ds_point_sampled_already
from src.metric import chamfer_distance_knn


@dataclass
class SetVaeConfig:
    # paths
    data_root: str = "data"
    ckpt_dir: str = "checkpoints_setvae"
    log_dir: str = "logs_setvae"

    # model
    num_points: int = 2048
    hidden_dim: int = 64
    z_dim: int = 16
    z_scales: str = "1,1,2,4,8,16,32"  # comma-separated, parsed before use
    n_mixtures: int = 4
    init_dim: int = 32
    num_heads: int = 4
    ln: bool = True
    dropout: float = 0.0
    slot_att: bool = True
    train_gmm: bool = True

    # training
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    beta: float = 1.0
    kl_warmup_epochs: int = 50
    free_bits: float = 0.5   # nats mínimos por camada (previne posterior collapse)

    # data
    val_split: float = 0.1
    num_workers: int = 4
    seed: int = 42

    # logging
    save_every: int = 5
    log_every: int = 50
    device: str = "cuda"
    amp: bool = False
    resume: int = 0


def parse_args():
    cfg = SetVaeConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=cfg.data_root)
    p.add_argument("--ckpt_dir", default=cfg.ckpt_dir)
    p.add_argument("--log_dir", default=cfg.log_dir)
    p.add_argument("--num_points", type=int, default=cfg.num_points)
    p.add_argument("--hidden_dim", type=int, default=cfg.hidden_dim)
    p.add_argument("--z_dim", type=int, default=cfg.z_dim)
    p.add_argument("--z_scales", default=cfg.z_scales)
    p.add_argument("--n_mixtures", type=int, default=cfg.n_mixtures)
    p.add_argument("--init_dim", type=int, default=cfg.init_dim)
    p.add_argument("--num_heads", type=int, default=cfg.num_heads)
    p.add_argument("--ln", type=lambda x: x.lower() != "false", default=cfg.ln)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--slot_att", type=lambda x: x.lower() != "false", default=cfg.slot_att)
    p.add_argument("--train_gmm", type=lambda x: x.lower() != "false", default=cfg.train_gmm)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--warmup_epochs", type=int, default=cfg.warmup_epochs)
    p.add_argument("--grad_clip", type=float, default=cfg.grad_clip)
    p.add_argument("--beta", type=float, default=cfg.beta)
    p.add_argument("--kl_warmup_epochs", type=int, default=cfg.kl_warmup_epochs)
    p.add_argument("--free_bits", type=float, default=cfg.free_bits)
    p.add_argument("--val_split", type=float, default=cfg.val_split)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--save_every", type=int, default=cfg.save_every)
    p.add_argument("--log_every", type=int, default=cfg.log_every)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--amp", action="store_true", default=cfg.amp)
    p.add_argument("--resume", type=int, default=cfg.resume)
    args = p.parse_args()
    # copy parsed args into dataclass
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


def warmup_lr(optimizer, step, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
        for g in optimizer.param_groups:
            g["lr"] = lr


def save_checkpoint(model, optimizer, epoch, cfg, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(cfg),
    }, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


def visualize_reconstructions(model, batch, writer, step, device, tag="recon"):
    model.eval()
    with torch.no_grad():
        pts = batch[0][:4].to(device)  # (4, N, C)
        xyz_out = model.reconstruct(pts)  # (4, N, 3)
        gt_xyz = normalize_pc(pts)[..., :3]
        cd, _, _ = chamfer_distance_knn(xyz_out, gt_xyz, reduce="mean")
        writer.add_scalar(f"{tag}/cd", cd.item(), step)
        # log point clouds as 3D scatter (add_mesh if available, else skip)
        try:
            # TensorBoard add_mesh expects (B, N, 3)
            writer.add_mesh(f"{tag}/gt", gt_xyz[:1], global_step=step)
            writer.add_mesh(f"{tag}/pred", xyz_out[:1], global_step=step)
        except Exception:
            pass
    model.train()
    return cd.item()


def main(cfg: SetVaeConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    # parse z_scales string
    z_scales = [int(x) for x in cfg.z_scales.split(",")]

    # dataset
    dataset = Ds_point_sampled_already(cfg.data_root, augment=True)
    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train:])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = SetVAE(
        input_dim=3,
        num_points=cfg.num_points,
        hidden_dim=cfg.hidden_dim,
        z_scales=z_scales,
        z_dim=cfg.z_dim,
        n_mixtures=cfg.n_mixtures,
        init_dim=cfg.init_dim,
        num_heads=cfg.num_heads,
        ln=cfg.ln,
        dropout=cfg.dropout,
        slot_att=cfg.slot_att,
        train_gmm=cfg.train_gmm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    start_epoch = 0
    if cfg.resume > 0:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{cfg.resume:04d}.pt")
        if os.path.exists(ckpt_path):
            start_epoch = load_checkpoint(ckpt_path, model, optimizer)
            print(f"Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SetVAE — {n_params/1e6:.2f}M parameters")
    print(f"z_scales={z_scales}, total latent dims={sum(z_scales)*cfg.z_dim}")

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        n_batches = 0

        for batch_idx, (pts, _) in enumerate(train_loader):
            pts = pts.to(device, non_blocking=True)  # (B, N, 6)
            # normaliza para o mesmo espaço que o decoder produz
            target_xyz = normalize_pc(pts)[..., :3]
            B = pts.shape[0]

            # lr warmup
            warmup_lr(optimizer, global_step, cfg.warmup_epochs * len(train_loader), cfg.lr)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                xyz_out, kls = model(pts)
                recon, _, _ = chamfer_distance_knn(xyz_out, target_xyz, reduce="mean")

                if len(kls) > 0:
                    # free bits: cada camada contribui no mínimo free_bits nats
                    # previne posterior collapse (posterior → prior → z inútil)
                    kl_free = [kl.clamp(min=cfg.free_bits) for kl in kls]
                    kl_total = torch.stack(kl_free, dim=1).sum(dim=1).mean()
                else:
                    kl_total = torch.tensor(0.0, device=device)

                beta_t = cfg.beta * min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
                loss = recon + beta_t * kl_total

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl_total.item()
            n_batches += 1

            if global_step % cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/recon", recon.item(), global_step)
                writer.add_scalar("train/kl", kl_total.item(), global_step)
                writer.add_scalar("train/beta_t", beta_t, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                # per-layer KL
                for i, kl_i in enumerate(kls):
                    writer.add_scalar(f"latent/kl_layer_{i}", kl_i.mean().item(), global_step)

            global_step += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        print(f"[{epoch+1:04d}/{cfg.epochs}] loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  beta={beta_t:.3f}")
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        writer.add_scalar("epoch/recon", avg_recon, epoch)
        writer.add_scalar("epoch/kl", avg_kl, epoch)

        # validation
        if (epoch + 1) % cfg.save_every == 0 or epoch == 0:
            model.eval()
            val_recon = 0.0
            val_n = 0
            with torch.no_grad():
                for pts, _ in val_loader:
                    pts = pts.to(device)
                    target_xyz = normalize_pc(pts)[..., :3]
                    xyz_out = model.reconstruct(pts)
                    cd, _, _ = chamfer_distance_knn(xyz_out, target_xyz, reduce="mean")
                    val_recon += cd.item() * pts.shape[0]
                    val_n += pts.shape[0]
            val_recon /= max(val_n, 1)
            writer.add_scalar("val/recon", val_recon, epoch)
            print(f"  val_recon={val_recon:.4f}")

            # visualize with first val batch
            val_iter = iter(val_loader)
            val_batch = next(val_iter)
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
            recon_cd = visualize_reconstructions(model, train_batch, writer, epoch, device, tag="recon")
            writer.add_scalar("val/recon_cd", recon_cd, epoch)

            # save checkpoint
            ckpt_path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
            #save_checkpoint(model, optimizer, epoch + 1, cfg, ckpt_path)
            print(f"  saved {ckpt_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
