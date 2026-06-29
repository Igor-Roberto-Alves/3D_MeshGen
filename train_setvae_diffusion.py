"""
train_setvae_diffusion.py
=========================
Stage 2 do pipeline SetVAE: treina um DDPM no espaço latente hierárquico achatado
z_flat ∈ ℝ^{sum(z_scales)*z_dim} extraído pelo SetVAE treinado.

Fluxo de geração:
  z_flat ~ DDPM → SetVAE.decode_latents(z_flat) → nuvem de pontos (B, N, 3)
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.dataset import Ds_point_sampled_already
from src.SetVAE import SetVAE
from src.Diffusion import LinearSchedule, sinusoidal_emb, MLPResBlock
from src.metric import chamfer_distance_knn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── Denoiser ──────────────────────────────────────────────────────────────

class SetVAEDenoiser(nn.Module):
    """
    ε-predictor MLP para z_flat ∈ ℝ^{z_flat_dim}.
    Opcionalmente condicionado em label de classe via embedding.
    """
    def __init__(self, z_flat_dim: int, hidden: int = 512, n_layers: int = 8,
                 num_classes: int = 0, class_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        time_dim = hidden

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim))

        cond_in = time_dim + (class_dim if num_classes > 0 else 0)
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes + 1, class_dim)  # +1 = null (CFG)
        self.cond_proj = nn.Linear(cond_in, time_dim)

        self.input_proj  = nn.Linear(z_flat_dim, hidden)
        self.blocks      = nn.ModuleList([MLPResBlock(hidden, time_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, z_flat_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, zt: torch.Tensor, t: torch.Tensor,
                class_idx: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = sinusoidal_emb(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        if self.num_classes > 0 and class_idx is not None:
            c_emb = self.class_emb(class_idx)
            cond  = self.cond_proj(torch.cat([t_emb, c_emb], dim=-1))
        else:
            cond = self.cond_proj(t_emb)
        h = self.input_proj(zt)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class DiffConfig:
    # paths
    vae_ckpt:       str   = "checkpoints_setvae/best.pt"
    ckpt_dir:       str   = "checkpoints_setvae_diff"
    log_dir:        str   = "logs_setvae_diff"
    data_root:      str   = "point_clouds"
    # denoiser
    hidden:         int   = 512
    n_layers:       int   = 8
    num_classes:    int   = 55   # 0 = uncondicional
    class_dim:      int   = 128
    cfg_dropout:    float = 0.1  # prob de dropar label (classifier-free guidance)
    guidance:       float = 3.0
    # schedule
    T:              int   = 1000
    # training
    epochs:         int   = 200
    batch_size:     int   = 32
    lr:             float = 2e-4
    weight_decay:   float = 1e-4
    warmup_epochs:  int   = 5
    grad_clip:      float = 1.0
    val_split:      float = 0.1
    num_workers:    int   = 4
    seed:           int   = 42
    save_every:     int   = 10
    log_every:      int   = 20
    vis_every:      int   = 20
    vis_n:          int   = 8
    device:         str   = "cuda"
    amp:            bool  = False
    resume:         int   = 0


# ── Helpers ────────────────────────────────────────────────────────────────

def diffusion_loss(denoiser, schedule, z_flat, class_idx, cfg_dropout, T, device):
    B = z_flat.shape[0]
    t  = torch.randint(0, T, (B,), device=device)
    eps = torch.randn_like(z_flat)
    zt  = schedule.q_sample(z_flat, t, eps)
    # classifier-free guidance: randomly drop class
    if cfg_dropout > 0 and denoiser.num_classes > 0:
        drop_mask = torch.rand(B, device=device) < cfg_dropout
        class_idx_in = class_idx.clone()
        class_idx_in[drop_mask] = denoiser.num_classes   # null token
    else:
        class_idx_in = class_idx
    eps_pred = denoiser(zt, t, class_idx_in if denoiser.num_classes > 0 else None)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def precompute_latents(vae: SetVAE, loader: DataLoader, device: torch.device):
    """Extract z_flat for all shapes. Returns (z_flat_all, labels_all) tensors."""
    zs, labels = [], []
    for points, cls_idx in loader:
        points = points.to(device)
        z = vae.encode_latents(points)   # (B, z_flat_dim)
        zs.append(z.cpu())
        labels.append(cls_idx)
    return torch.cat(zs, 0), torch.cat(labels, 0)


def load_vae(ckpt_path: str, device: torch.device) -> SetVAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})
    vae  = SetVAE(
        input_dim  = cfg.get("input_dim", 3),
        num_points = cfg.get("num_points", 2048),
        hidden_dim = cfg.get("hidden_dim", 64),
        z_scales   = cfg.get("z_scales", [1, 1, 2, 4, 8, 16, 32]),
        z_dim      = cfg.get("z_dim", 16),
        n_mixtures = cfg.get("n_mixtures", 4),
        init_dim   = cfg.get("init_dim", 32),
        num_heads  = cfg.get("num_heads", 4),
        ln         = cfg.get("ln", True),
        dropout    = cfg.get("dropout", 0.0),
        slot_att   = cfg.get("slot_att", True),
        train_gmm  = cfg.get("train_gmm", True),
    )
    vae.load_state_dict(ckpt["model"])
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ── Training ───────────────────────────────────────────────────────────────

def main(cfg: DiffConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    # ── Load VAE ─────────────────────────────────────────────────────────
    log.info(f"Carregando VAE de {cfg.vae_ckpt}")
    vae = load_vae(cfg.vae_ckpt, device)
    z_flat_dim = vae.z_flat_dim
    log.info(f"z_flat_dim = {z_flat_dim}  (= {sum(vae.z_scales)} × {vae.z_dim})")

    # ── Dataset + precompute latents ─────────────────────────────────────
    full_ds  = Ds_point_sampled_already(cfg.data_root, augment=False)
    n_val    = max(1, int(len(full_ds) * cfg.val_split))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    log.info("Pré-computando latentes do VAE para treino…")
    z_train, y_train = precompute_latents(vae, train_loader, device)
    z_val,   y_val   = precompute_latents(vae, val_loader,   device)

    # normalização
    z_mean = z_train.mean(0, keepdim=True)
    z_std  = z_train.std(0, keepdim=True).clamp(min=1e-6)
    z_train_n = (z_train - z_mean) / z_std
    z_val_n   = (z_val   - z_mean) / z_std

    train_ld = DataLoader(TensorDataset(z_train_n, y_train),
                          batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_ld   = DataLoader(TensorDataset(z_val_n, y_val),
                          batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # ── Denoiser + schedule ───────────────────────────────────────────────
    denoiser = SetVAEDenoiser(
        z_flat_dim  = z_flat_dim,
        hidden      = cfg.hidden,
        n_layers    = cfg.n_layers,
        num_classes = cfg.num_classes,
        class_dim   = cfg.class_dim,
    ).to(device)
    schedule = LinearSchedule(T=cfg.T).to(device)

    n_params = sum(p.numel() for p in denoiser.parameters())
    log.info(f"Denoiser: {n_params/1e6:.2f}M parâmetros")

    opt = AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.lr * 0.01)
    scheduler = SequentialLR(opt, [warmup, cosine], milestones=[cfg.warmup_epochs])

    start_epoch = 0
    best_val    = float("inf")

    if cfg.resume:
        ckpt_path = Path(cfg.ckpt_dir) / f"epoch_{cfg.resume:04d}.pt"
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device)
            denoiser.load_state_dict(ck["denoiser"])
            opt.load_state_dict(ck["opt"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"] + 1
            best_val    = ck.get("best_val", float("inf"))
            log.info(f"Retomando do epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # ── Loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        denoiser.train()
        losses = []
        for step, (z_batch, cls_batch) in enumerate(train_ld):
            z_batch   = z_batch.to(device)
            cls_batch = cls_batch.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                loss = diffusion_loss(denoiser, schedule, z_batch, cls_batch,
                                      cfg.cfg_dropout, cfg.T, device)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if (step + 1) % cfg.log_every == 0:
                log.info(f"[E{epoch:03d} S{step+1}] loss={np.mean(losses[-cfg.log_every:]):.4f}")

        scheduler.step()
        train_loss = np.mean(losses)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)

        # ── Validação ─────────────────────────────────────────────────────
        denoiser.eval()
        val_losses = []
        with torch.no_grad():
            for z_batch, cls_batch in val_ld:
                z_batch   = z_batch.to(device)
                cls_batch = cls_batch.to(device)
                loss = diffusion_loss(denoiser, schedule, z_batch, cls_batch,
                                      0.0, cfg.T, device)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        writer.add_scalar("val/loss", val_loss, epoch)

        log.info(f"[E{epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ── Visualização / CD ─────────────────────────────────────────────
        if epoch % cfg.vis_every == 0:
            with torch.no_grad():
                # gera amostras via DDPM (classe 0 como referência)
                cls_vis = torch.zeros(cfg.vis_n, dtype=torch.long, device=device)
                if denoiser.num_classes > 0:
                    uncond = torch.full_like(cls_vis, denoiser.num_classes)
                    z_gen  = schedule.sample(denoiser, (z_flat_dim,),
                                             condition=cls_vis, uncond=uncond,
                                             guidance=cfg.guidance, device=device)
                else:
                    # cls_vis passado como dummy; denoiser ignora quando num_classes=0
                    z_gen = schedule.sample(denoiser, (z_flat_dim,),
                                            condition=cls_vis, device=device)

                z_gen_denorm = z_gen * z_std.to(device) + z_mean.to(device)
                xyz_gen = vae.decode_latents(z_gen_denorm)      # (B, N, 3)

                # CD contra val real
                val_pts = next(iter(val_ld))[0][:cfg.vis_n].to(device)
                val_pts_xyz = val_pts[..., :3]
                cd, _, _ = chamfer_distance_knn(xyz_gen, val_pts_xyz)
                writer.add_scalar("val/gen_cd", cd.item(), epoch)
                log.info(f"  [vis] gen_cd={cd.item():.5f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        state = {
            "epoch": epoch, "denoiser": denoiser.state_dict(),
            "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "config": asdict(cfg),
            "z_mean": z_mean, "z_std": z_std,
            "z_flat_dim": z_flat_dim,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, Path(cfg.ckpt_dir) / "best.pt")
            log.info(f"  Melhor val_loss={best_val:.4f} → salvo.")
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(state, Path(cfg.ckpt_dir) / f"epoch_{epoch:04d}.pt")

    writer.close()
    log.info("Treino concluído.")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    cfg = DiffConfig()
    p = argparse.ArgumentParser()
    for k, v in asdict(cfg).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    return DiffConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
