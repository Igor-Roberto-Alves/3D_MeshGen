"""
train.py  –  Auditable Training Loop with Full TensorBoard Integration
=======================================================================

What's new vs. the original ``train.py``:
──────────────────────────────────────────
① TensorBoard writer logs *every* scalar, histogram, and point-cloud image.
② Three-phase training schedule:
      Phase 0  (warm-up)       — β-VAE only (no GAN), β annealed from 0 → target
      Phase 1  (GAN intro)     — VAE+GAN, discriminator trained alternately
      Phase 2  (full / stable) — same as phase 1 with gradient penalty (WGAN-GP)
③ Gradient-penalty toggle (``use_gp``).
④ T-Net orthogonality reg loss added when the discriminator's T-Net is active.
⑤ Learning-rate schedulers (cosine with warm restart for G, step for D).
⑥ Gradient clipping on the generator/encoder.
⑦ Per-epoch histogram of latent ``z``, ``mu``, ``logvar``.
⑧ Best-model checkpoint saved separately (lowest validation CD).
⑨ ``--resume`` flag to continue from a checkpoint.
⑩ Clean ``argparse`` interface with sensible defaults.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Vae import DualBranchPointVAE
from src.GAN import PointNetDiscriminator
from src.dataset import Ds_point_sampled, Ds_point_model, Ds_point_sampled_already


# ─────────────────────────────────────────────────────────────────────────────
# Constants / schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def beta_schedule(epoch: int, warmup_epochs: int, target_beta: float) -> float:
    """Linear β annealing from 0 to target_beta over warmup_epochs."""
    if warmup_epochs == 0:
        return target_beta
    return min(target_beta, target_beta * epoch / warmup_epochs)


def get_phase(epoch: int, warmup: int, gan_start: int) -> int:
    """
    Returns 0 (VAE only), 1 (VAE+GAN, no GP), or 2 (VAE+GAN with GP).
    GP kicks in 10 epochs after GAN intro to let D stabilise first.
    """
    if epoch < warmup:
        return 0
    if epoch < gan_start + 10:
        return 1
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# Point-cloud  → RGB image  (for TensorBoard add_image)
# ─────────────────────────────────────────────────────────────────────────────

def pcd_to_image(pts: torch.Tensor, img_size: int = 128) -> torch.Tensor:
    """
    Project a single point cloud (N, 3+) to a top-down 2-D grid image.
    Returns a (3, H, W) float tensor in [0, 1] suitable for TensorBoard.
    Normalises XY coordinates to [0, img_size) and counts occupancy per pixel.
    """
    xyz = pts[:, :3].detach().cpu()
    # Normalise to [0, img_size)
    xyz = xyz - xyz.min(dim=0).values
    span = xyz.max(dim=0).values.clamp(min=1e-6)
    xyz = xyz / span * (img_size - 1)

    canvas = torch.zeros(img_size, img_size)
    ix = xyz[:, 0].long().clamp(0, img_size - 1)
    iy = xyz[:, 1].long().clamp(0, img_size - 1)
    canvas.index_put_((iy, ix), torch.ones(len(ix)), accumulate=True)
    canvas = (canvas / canvas.max().clamp(min=1e-6)).unsqueeze(0)   # (1,H,W)
    return canvas.expand(3, -1, -1)                                  # (3,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# Save .ply helper
# ─────────────────────────────────────────────────────────────────────────────

def save_ply(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(arr[:, :3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:, 3:6])
    o3d.io.write_point_cloud(path, pcd)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    recon_dir = run_dir / "reconstructions"
    tb_dir = run_dir / "tensorboard"
    for d in (ckpt_dir, recon_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[INFO] TensorBoard dir → {tb_dir}")
    print(f"[INFO] Run:  tensorboard --logdir {tb_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────
    if args.dataset_cached:
        full_ds = Ds_point_sampled_already()
    else:
        full_ds = Ds_point_sampled(Ds_point_model())

    val_size  = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"[INFO] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = DualBranchPointVAE(
        d_model=args.d_model,
        latent_dim=args.latent_dim,
        n_out=args.n_out,
        enc_depth=args.enc_depth,

        n_heads=args.n_heads,
        beta=args.beta_target,
        lambda_adv=args.lambda_adv,
        disc_base_ch=args.disc_base_ch,
    ).to(device)
    model.report_parameters()

    # ── Optimisers ────────────────────────────────────────────────────────
    # Separate parameter groups: encoder + decoder (generator) vs discriminator
    gen_params  = (
        list(model.hier_enc.parameters())
        + list(model.glob_enc.parameters())
        + list(model.fusion.parameters())
        + list(model.bottleneck.parameters())
        + list(model.decoder.parameters())
    )
    disc_params = list(model.discriminator.parameters())

    opt_G = torch.optim.AdamW(gen_params,  lr=args.lr_g, weight_decay=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.AdamW(disc_params, lr=args.lr_d, weight_decay=1e-4, betas=(0.5, 0.999))

    sched_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_G, T_0=args.cosine_t0, T_mult=2, eta_min=1e-6
    )
    sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=50, gamma=0.5)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_cd = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt_G.load_state_dict(ckpt["opt_G_state"])
        opt_D.load_state_dict(ckpt["opt_D_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_cd = ckpt.get("best_val_cd", float("inf"))
        print(f"[INFO] Resumed from epoch {start_epoch}  (best CD={best_val_cd:.6f})")

    # ── Log graph (once) ──────────────────────────────────────────────────
    try:
        dummy = torch.zeros(1, 64, 6, device=device)
        writer.add_graph(model.discriminator, dummy)
    except Exception:
        pass   # graph tracing sometimes fails with spectral norm — non-fatal

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        phase       = get_phase(epoch, args.warmup_epochs, args.gan_start_epoch)
        beta_now    = beta_schedule(epoch, args.warmup_epochs, args.beta_target)
        model.beta  = beta_now
        use_gp      = (phase == 2) and args.use_gp

        # Accumulators
        acc: dict[str, float] = {
            "loss_G": 0.0, "cd": 0.0, "kl": 0.0, "normal_loss": 0.0,
            "loss_D": 0.0, "loss_D_real": 0.0, "loss_D_fake": 0.0,
            "gp": 0.0,
        }
        n_batches = 0

        last_xyz = last_recon = last_class = None

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}  phase={phase}")

        for batch in loop:
            class_name, pcd = batch
            xyz = pcd.to(device)          # (B, N, 6)
            B   = xyz.size(0)

            # ── Step 1 : Discriminator update  (phases 1 and 2 only) ──────
            if phase >= 1:
                model.unfreeze_D()
                opt_D.zero_grad()

                with torch.no_grad():
                    fake_out = model(xyz)
                fake_pts = fake_out["recon"].detach()    # (B, N, 6)

                d_losses = model.loss_discriminator(xyz, fake_pts)
                loss_D   = d_losses["loss_D"]

                # Gradient penalty
                if use_gp:
                    gp = PointNetDiscriminator.compute_gradient_penalty(
                        model.discriminator, xyz, fake_pts,
                        lambda_gp=args.lambda_gp,
                    )
                    loss_D = loss_D + gp
                    acc["gp"] += gp.item()

        
                

                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, max_norm=1.0)
                opt_D.step()

                acc["loss_D"]      += d_losses["loss_D"].item()
                acc["loss_D_real"] += d_losses["loss_D_real"].item()
                acc["loss_D_fake"] += d_losses["loss_D_fake"].item()

            # ── Step 2 : Generator / VAE update ──────────────────────────
            model.freeze_D()
            opt_G.zero_grad()

            out = model(xyz)

            if phase == 0:
                g_losses = model.loss(out, xyz)         # pure β-VAE (chamfer)
                loss_G   = g_losses["total"]
            else:
                g_losses = model.loss_generator(out, xyz)  # sinkhorn + GAN
                loss_G   = g_losses["total"]

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=args.grad_clip)
            opt_G.step()

            def _item(v) -> float:
                return v.item() if isinstance(v, torch.Tensor) else float(v)

            acc["loss_G"]      += _item(loss_G)
            acc["cd"]          += _item(g_losses.get("cd",          0.0))
            acc["kl"]          += _item(g_losses.get("kl",          0.0))
            acc["normal_loss"] += _item(g_losses.get("normal_loss", 0.0))

            # ── Unfreeze D for next iteration ─────────────────────────────
            model.unfreeze_D()

            n_batches += 1

            loop.set_postfix(
                G=f"{_item(loss_G):.4f}",
                cd=f"{_item(g_losses.get('cd', 0.0)):.4f}",
                β=f"{beta_now:.5f}",
                ph=phase,
            )

            last_xyz   = xyz
            last_recon = out["recon"]
            last_class = class_name[0]

        # ── Averages ──────────────────────────────────────────────────────
        avg = {k: v / max(n_batches, 1) for k, v in acc.items()}
        global_step = epoch + 1

        # ── TensorBoard: scalars ──────────────────────────────────────────
        writer.add_scalar("train/loss_G",      avg["loss_G"],      global_step)
        writer.add_scalar("train/cd",          avg["cd"],          global_step)
        writer.add_scalar("train/kl",          avg["kl"],          global_step)
        writer.add_scalar("train/normal_loss", avg["normal_loss"], global_step)
        writer.add_scalar("train/loss_D",      avg["loss_D"],      global_step)
        writer.add_scalar("train/loss_D_real", avg["loss_D_real"], global_step)
        writer.add_scalar("train/loss_D_fake", avg["loss_D_fake"], global_step)
        writer.add_scalar("train/grad_penalty",avg["gp"],          global_step)

        writer.add_scalar("train/beta",        beta_now,           global_step)
        writer.add_scalar("train/phase",       float(phase),       global_step)
        writer.add_scalar("lr/generator",
                          opt_G.param_groups[0]["lr"], global_step)
        writer.add_scalar("lr/discriminator",
                          opt_D.param_groups[0]["lr"], global_step)

        # ── TensorBoard: latent histograms ────────────────────────────────

        if last_xyz is not None:
            with torch.no_grad():
                z, mu, logvar, style = model.encode(last_xyz)
            
            # Validação para impedir que tensores vazios ou com NaN/Inf quebrem o TensorBoard
            if mu.numel() > 0 and not (torch.isnan(mu).any() or torch.isinf(mu).any()):
                writer.add_histogram("latent/mu",     mu.detach().cpu().numpy(),     global_step)
                writer.add_histogram("latent/logvar", logvar.detach().cpu().numpy(), global_step)
                writer.add_histogram("latent/std",    (0.5 * logvar).exp().detach().cpu().numpy(), global_step)
            else:
                print(f"⚠️ [Aviso] Pulando histogramas do latent space no epoch {global_step} (Valores inválidos ou NaN detectados).")

        # ── TensorBoard: point-cloud images ───────────────────────────────
        if last_xyz is not None and last_recon is not None:
            img_orig  = pcd_to_image(last_xyz[0])     # (3, H, W)
            img_recon = pcd_to_image(last_recon[0])
            writer.add_image("pointcloud/original",     img_orig,  global_step)
            writer.add_image("pointcloud/reconstructed",img_recon, global_step)

        # ── TensorBoard: weight histograms (every 10 epochs) ─────────────
        if (epoch + 1) % 10 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel() > 0:
                    # Garante que não há NaN antes de plotar os pesos
                    if not torch.isnan(param).any():
                        writer.add_histogram(f"weights/{name}", param.detach().cpu().numpy(), global_step)
                    if param.grad is not None and not torch.isnan(param.grad).any():
                        writer.add_histogram(f"grads/{name}", param.grad.detach().cpu().numpy(), global_step)

        # ── Validation ────────────────────────────────────────────────────
        val_cd = _validate(model, val_loader, device)
        writer.add_scalar("val/cd", val_cd, global_step)

        is_best = val_cd < best_val_cd
        if is_best:
            best_val_cd = val_cd

        print(
            f"Epoch {global_step}/{args.num_epochs}  "
            f"G={avg['loss_G']:.4f}  CD={avg['cd']:.4f}  "
            f"KL={avg['kl']:.4f}  NL={avg['normal_loss']:.4f}  "
            f"D={avg['loss_D']:.4f}  ValCD={val_cd:.4f}"
            + ("  ← best" if is_best else "")
        )

        # ── Save .ply sample ──────────────────────────────────────────────
        if last_xyz is not None:
            save_ply(
                last_xyz[0],
                str(recon_dir / f"ep{global_step}_{last_class}_ORIG.ply"),
            )
            save_ply(
                last_recon[0],
                str(recon_dir / f"ep{global_step}_{last_class}_RECON.ply"),
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt_payload = {
            "epoch":        global_step,
            "model_state":  model.state_dict(),
            "opt_G_state":  opt_G.state_dict(),
            "opt_D_state":  opt_D.state_dict(),
            "best_val_cd":  best_val_cd,
            "args":         vars(args),
        }
        torch.save(
            ckpt_payload,
            str(ckpt_dir / f"epoch_{global_step:04d}.pt"),
        )
        if is_best:
            torch.save(ckpt_payload, str(ckpt_dir / "best_model.pt"))
            print(f"  [✓] Best model saved  (val CD={best_val_cd:.6f})")

        # ── LR schedulers ─────────────────────────────────────────────────
        sched_G.step()
        sched_D.step()

    writer.close()
    print("[INFO] Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
from src.metric import *

def _validate(
    model: DualBranchPointVAE,
    loader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()

    total_loss = 0.0
    n = 0

    with torch.no_grad():

        for _, pcd in loader:

            xyz = pcd.to(device)

            out = model(xyz)

            pred_xyz = out["recon"][..., :3]
            gt_xyz = xyz[..., :3]

            cd_loss, _ = chamfer_distance(
                out["recon"],
                xyz
            )

            emd_loss, normal_loss = earth_movers_distance_sinkhorn(
                out["recon"],
                xyz
            )

            val_loss = cd_loss + 0.05 * emd_loss

            total_loss += val_loss.item()

            n += 1

    model.train()

    return total_loss / max(n, 1)





# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auditable DualBranchPointVAE training with TensorBoard"
    )

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset_cached",
                   help="Use Ds_point_sampled_already (pre-processed dataset)", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of data reserved for validation (default 0.10)")
    p.add_argument("--num_workers", type=int, default=4)

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--d_model",      type=int,   default=384)
    p.add_argument("--latent_dim",   type=int,   default=512)
    p.add_argument("--n_out",        type=int,   default=2048)
    p.add_argument("--enc_depth",    type=int,   default=8)
    p.add_argument("--dec_depth",    type=int,   default=4)
    p.add_argument("--n_heads",      type=int,   default=6)
    p.add_argument("--disc_base_ch", type=int,   default=64)

    # ── training schedule ─────────────────────────────────────────────────
    p.add_argument("--num_epochs",      type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr_g",            type=float, default=1e-3,
                   help="Generator / encoder learning rate")
    p.add_argument("--lr_d",            type=float, default=4e-4,
                   help="Discriminator learning rate (usually lower than G)")
    p.add_argument("--cosine_t0",       type=int,   default=50,
                   help="CosineAnnealingWarmRestarts T_0 for the G scheduler")
    p.add_argument("--grad_clip",       type=float, default=1.0,
                   help="Max gradient norm for generator/encoder (0 = disabled)")
    p.add_argument("--warmup_epochs",   type=int,   default=20,
                   help="Pure β-VAE warm-up before introducing the GAN")
    p.add_argument("--gan_start_epoch", type=int,   default=20,
                   help="Epoch at which the adversarial loss kicks in")

    # ── loss coefficients ─────────────────────────────────────────────────
    p.add_argument("--beta_target",  type=float, default=1e-3,
                   help="Target β for KL (annealed from 0 during warm-up)")
    p.add_argument("--lambda_adv",   type=float, default=0.1)
    p.add_argument("--lambda_gp",    type=float, default=10.0,
                   help="Gradient penalty coefficient (WGAN-GP)")
    p.add_argument("--use_gp",       action="store_true",
                   help="Enable WGAN-GP gradient penalty (phase 2 onwards)")

    # ── I/O ───────────────────────────────────────────────────────────────
    p.add_argument("--run_dir", type=str, default="runs/default",
                   help="Root directory for checkpoints, TensorBoard, and PLY saves")
    p.add_argument("--resume",  type=str, default=None,
                   help="Path to a checkpoint .pt file to resume from")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    train(parse_args())