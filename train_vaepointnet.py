"""
train_vaepointnet.py
--------------------
Training script for VaePointnet on ShapeNet PLY files in point_clouds/.

TensorBoard logs written to runs/<exp_name>/:
  Scalars  — loss, CD, KL, beta, lr (every epoch)
  Mesh     — Recon/original_vs_recon : original (blue) + recon (orange)
           — Sample/from_prior       : shapes sampled from N(0,I) (green)
  Latent   — Latent/mu_hist, mu_std_per_dim, active_dims

Launch viewer:
  tensorboard --logdir runs

Quick-start examples
--------------------
  # Exp 0 — baseline sanity
  python train_vaepointnet.py --exp_name exp0_baseline

  # Exp 1c — higher KL weight
  python train_vaepointnet.py --exp_name exp1c_beta001 --beta 0.01

  # Exp 2 — KL annealing (recommended first real experiment)
  python train_vaepointnet.py --exp_name exp2_anneal --beta 0.01 --kl_anneal_epochs 100

  # Exp 3b — smaller latent
  python train_vaepointnet.py --exp_name exp3b_latent128 --latent_dim 128

  # Exp 4 — FoldingNet decoder
  python train_vaepointnet.py --exp_name exp4_folding --decoder_type folding

  # Exp 5 — with normals (6-D input)
  python train_vaepointnet.py --exp_name exp5_normals --in_dim 6

  # Exp 6 — multi-category (airplanes + cars)
  python train_vaepointnet.py --exp_name exp6_all --category all

  # Exp 7 — CD converges but shapes look wrong: blend in EMD
  python train_vaepointnet.py --exp_name exp7_emd --recon_loss both --emd_weight 0.5

  # Exp 8 — same symptom, more capacity (bigger latent + folding decoder)
  python train_vaepointnet.py --exp_name exp8_cap --latent_dim 512 --global_dim 2048 \
      --decoder_type folding

  # Resume from last checkpoint
  python train_vaepointnet.py --exp_name exp2_anneal --resume
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from EncoderPointnet import EncoderPointnet, tnet_regularization
from DecoderPointnet  import DecoderMLP, DecoderFolding
from Vaepointnet      import VaePointnet
from src.metric       import chamfer_distance_knn, kl_divergence, emd_approx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_IDS = {"airplane": "02691156", "car": "02958343", "all": None}


def _load_ply(path: Path, in_dim: int) -> torch.Tensor:
    """Load one PLY, centre + scale to unit sphere, return [N, in_dim] float32."""
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)

    xyz -= xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz /= scale

    if in_dim == 6:
        nrm = np.asarray(pcd.normals, dtype=np.float32)
        if nrm.shape[0] != xyz.shape[0]:
            pcd.estimate_normals()
            nrm = np.asarray(pcd.normals, dtype=np.float32)
        pts = np.concatenate([xyz, nrm], axis=1)
    else:
        pts = xyz

    return torch.from_numpy(pts)


def _random_rotation() -> torch.Tensor:
    """Uniform random rotation matrix in SO(3) via QR decomposition."""
    q, r = torch.linalg.qr(torch.randn(3, 3))
    # Ensure det(q) = +1 (proper rotation, not reflection)
    q *= torch.sign(torch.linalg.det(q))
    return q                                                  # [3, 3]


class PointCloudDataset(Dataset):
    def __init__(self, data_root: str, category: str = "airplane",
                 in_dim: int = 3, num_points: int = 2048, augment: bool = True):
        self.in_dim     = in_dim
        self.num_points = num_points
        self.augment    = augment
        files           = []

        cat_id = CATEGORY_IDS.get(category)
        root   = Path(data_root)

        for f in sorted(root.glob("*.ply")):
            if cat_id is None or f.name.startswith(cat_id):
                files.append(f)

        if not files:
            raise RuntimeError(
                f"No PLY files found in '{data_root}' for category='{category}'. "
                f"Expected prefix '{cat_id}'."
            )

        # Pre-cache all point clouds in RAM — 7559 × 2048 × 6 × 4B ≈ 376 MB max.
        # Eliminates open3d overhead from the training loop entirely.
        print(f"Loading {len(files)} PLY files into RAM ...", flush=True)
        self.data = [_load_ply(f, in_dim) for f in files]
        print(f"Dataset ready: {len(self.data)} files | category={category} | in_dim={in_dim}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pts = self.data[idx]                                  # [N, in_dim]  (already normalised)
        N   = pts.shape[0]

        # Random point subsample (different subset every epoch)
        if N >= self.num_points:
            perm = torch.randperm(N)[:self.num_points]
            pts  = pts[perm]
        else:
            rep = (self.num_points // N) + 1
            pts = pts.repeat(rep, 1)[:self.num_points]

        # SO(3) random rotation augmentation
        if self.augment:
            R   = _random_rotation()                          # [3, 3]
            xyz = pts[:, :3] @ R.T                           # [N, 3]
            if self.in_dim == 6:
                nrm = pts[:, 3:] @ R.T                       # normals rotate the same way
                pts = torch.cat([xyz, nrm], dim=1)
            else:
                pts = xyz

        return pts                                            # [num_points, in_dim]


# ─────────────────────────────────────────────────────────────────────────────
# KL annealing
# ─────────────────────────────────────────────────────────────────────────────

def kl_weight(epoch: int, beta: float, anneal_epochs: int) -> float:
    """Linear warm-up 0 → beta over `anneal_epochs`, then constant."""
    if anneal_epochs <= 0:
        return beta
    return beta * min(1.0, epoch / max(anneal_epochs, 1))


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid_colors(batch: int, n_pts: int, r: int, g: int, b: int) -> torch.Tensor:
    """Return [batch, n_pts, 3] uint8 filled with a single RGB colour."""
    c = torch.tensor([r, g, b], dtype=torch.uint8)
    return c.view(1, 1, 3).expand(batch, n_pts, 3).contiguous()


@torch.no_grad()
def log_visuals(writer: SummaryWriter, model: VaePointnet,
                val_loader: DataLoader, device, epoch: int, n_show: int = 4):
    """
    Write geometry and latent statistics to TensorBoard.

    Recon/original_vs_recon
        Original shifted left (blue) + reconstruction shifted right (orange).
        Lets you spot mode collapse and sharpness at a glance.

    Sample/from_prior
        Shapes sampled from the prior p(z) = N(0, I). Should look like
        plausible members of the category if the posterior is well-regularised.

    Latent/mu_hist
        Histogram of all mu values across the val set. Should be ~N(0,1) for
        a well-trained VAE.

    Latent/mu_std_per_dim
        Standard deviation of mu per latent dimension. Dead dimensions have
        std ≈ 0; the number of live dims tells you the true rank of the latent.

    Latent/active_dims
        Count of dimensions with std > 0.1 (a scalar, easy to track over time).
    """
    model.eval()

    # Grab one fixed mini-batch from val for recon visualisation
    fixed_batch = next(iter(val_loader))[:n_show].to(device)  # [n_show, N, in_dim]
    x   = fixed_batch.permute(0, 2, 1)                         # [n_show, in_dim, N]
    xyz = fixed_batch[:, :, :3]                                 # [n_show, N, 3]

    recon, _, _, _ = model(x)
    Nr = min(recon.shape[1], xyz.shape[1])                      # handle folding decoder
    recon_xyz = recon[:, :Nr, :]

    B, N = n_show, xyz.shape[1]

    # Shift so original and recon sit side-by-side in the 3-D viewer
    orig_v  = xyz.clone();          orig_v[..., 0]  -= 1.2
    recon_v = recon_xyz.clone();    recon_v[..., 0] += 1.2
    verts   = torch.cat([orig_v, recon_v], dim=1)               # [B, N+Nr, 3]
    colors  = torch.cat([
        _solid_colors(B, N,  100, 149, 237),   # blue  — original
        _solid_colors(B, Nr, 255, 140,   0),   # orange — reconstruction
    ], dim=1)
    writer.add_mesh("Recon/original_vs_recon", verts, colors=colors, global_step=epoch)

    # Prior samples
    samples = model.sample(n_show, device)                      # [n_show, N_out, 3]
    Ns = samples.shape[1]
    writer.add_mesh("Sample/from_prior", samples,
                    colors=_solid_colors(n_show, Ns, 50, 205, 50),
                    global_step=epoch)

    # Collect mu over the full val set for latent statistics
    all_mu = []
    for pts in val_loader:
        mu, _, _ = model.encoder(pts.to(device).permute(0, 2, 1))
        all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)                           # [N_val, latent_dim]

    mu_std = all_mu.std(dim=0)
    writer.add_histogram("Latent/mu_hist",        all_mu.flatten(), global_step=epoch)
    writer.add_histogram("Latent/mu_std_per_dim", mu_std,           global_step=epoch)
    writer.add_scalar("Latent/active_dims",
                      (mu_std > 0.1).float().sum().item(),          global_step=epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction loss (Chamfer, EMD, or a blend of both)
# ─────────────────────────────────────────────────────────────────────────────

def compute_recon_loss(recon: torch.Tensor, xyz: torch.Tensor, recon_loss: str,
                        emd_weight: float, emd_iters: int, emd_n_subsample: int):
    """
    Returns (loss_term, cd) where `loss_term` is what gets optimised and `cd`
    is the plain Chamfer distance, always computed for logging / checkpoint
    selection so runs stay comparable across --recon_loss settings.
    """
    cd, _, _ = chamfer_distance_knn(recon, xyz)

    if recon_loss == "chamfer":
        return cd, cd

    # EMD requires a bijection, so pred and target must have equal point counts
    # (the folding decoder can output slightly more points than requested).
    target = xyz
    if recon.shape[1] != xyz.shape[1]:
        idx    = torch.randperm(xyz.shape[1], device=xyz.device)[:recon.shape[1]]
        target = xyz[:, idx, :]

    emd = emd_approx(recon, target, n_iters=emd_iters, n_subsample=emd_n_subsample)

    if recon_loss == "emd":
        return emd, cd
    # "both": blend of Chamfer (global structure, cheap) and EMD (uniform
    # coverage, sharper detail) — fixes the many-to-one matching that lets
    # plain Chamfer settle on a locally-clumped, generically-shaped decode.
    return (1 - emd_weight) * cd + emd_weight * emd, cd


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, beta_cur, free_bits,
              tnet_reg_weight, device, train: bool,
              recon_loss: str = "chamfer", emd_weight: float = 0.5,
              emd_iters: int = 15, emd_n_subsample: int = 512):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_sum = cd_sum = kl_sum = 0.0
    n_batches = 0

    with ctx:
        for pts in loader:
            pts = pts.to(device)
            x   = pts.permute(0, 2, 1)              # [B, in_dim, N]
            xyz = pts[:, :, :3]                      # [B, N, 3] — target

            recon, mu, logvar, A_feat = model(x)

            recon_term, cd = compute_recon_loss(
                recon, xyz, recon_loss, emd_weight, emd_iters, emd_n_subsample
            )
            kl        = kl_divergence(mu, logvar, free_bits=free_bits)
            t_reg     = tnet_regularization(A_feat) if tnet_reg_weight > 0 else 0.0
            loss      = recon_term + beta_cur * kl + tnet_reg_weight * t_reg

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_sum += loss.item()
            cd_sum    += cd.item()
            kl_sum    += kl.item()
            n_batches += 1

    n = max(n_batches, 1)
    return total_sum / n, cd_sum / n, kl_sum / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   default="point_clouds")
    p.add_argument("--category",    default="airplane", choices=["airplane", "car", "all"])
    p.add_argument("--in_dim",      type=int, default=3, choices=[3, 6])
    p.add_argument("--num_points",  type=int, default=2048)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--no_augment",  action="store_true",
                   help="Disable SO(3) rotation augmentation (useful for debugging)")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--latent_dim",   type=int, default=256)
    p.add_argument("--global_dim",   type=int, default=1024)
    p.add_argument("--decoder_type", default="mlp", choices=["mlp", "folding"])

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",           type=int,   default=300)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--beta",             type=float, default=0.001)
    p.add_argument("--kl_anneal_epochs", type=int,   default=0)
    p.add_argument("--free_bits",        type=float, default=0.5)
    p.add_argument("--tnet_reg",         type=float, default=0.001)
    p.add_argument("--recon_loss",       default="chamfer", choices=["chamfer", "emd", "both"],
                   help="'both' blends Chamfer + EMD (see --emd_weight); "
                        "recommended when CD converges but shapes still look wrong")
    p.add_argument("--emd_weight",       type=float, default=0.5,
                   help="Weight of EMD in the blend when --recon_loss both (0=pure CD, 1=pure EMD)")
    p.add_argument("--emd_iters",        type=int,   default=15,
                   help="Sinkhorn iterations for the EMD approximation")
    p.add_argument("--emd_n_subsample",  type=int,   default=512,
                   help="Subsample points before EMD's O(N^2) Sinkhorn cost matrix "
                        "(0 = use all num_points, slower and more memory)")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--exp_name",  default="exp")
    p.add_argument("--ckpt_dir",  default="ckpt_pointvae")
    p.add_argument("--tb_dir",    default="runs",
                   help="Root dir for TensorBoard event files (runs/<exp_name>/)")
    p.add_argument("--vis_every", type=int, default=10,
                   help="Log geometry and latent stats to TB every N epochs")
    p.add_argument("--n_show",    type=int, default=4,
                   help="Number of point clouds to visualise in TensorBoard")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=42)

    args = p.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    # Load once into RAM, then build two lightweight views with different augment flags.
    # Val is never augmented so the Chamfer metric is comparable across runs.
    augment = not args.no_augment
    full_ds = PointCloudDataset(args.data_root, args.category, args.in_dim, args.num_points,
                                augment=False)

    g = torch.Generator().manual_seed(args.seed)
    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    all_idx = torch.randperm(len(full_ds), generator=g).tolist()
    train_idx, val_idx = all_idx[:n_train], all_idx[n_train:]

    class _View(Dataset):
        """Index view over a PointCloudDataset with an independent augment flag."""
        def __init__(self, base: PointCloudDataset, indices: list, aug: bool):
            self.data      = [base.data[i] for i in indices]   # shallow copy of tensors
            self.num_points = base.num_points
            self.in_dim     = base.in_dim
            self.augment    = aug
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            pts = self.data[i]
            N   = pts.shape[0]
            if N >= self.num_points:
                pts = pts[torch.randperm(N)[:self.num_points]]
            else:
                pts = pts.repeat((self.num_points // N) + 1, 1)[:self.num_points]
            if self.augment:
                R   = _random_rotation()
                xyz = pts[:, :3] @ R.T
                pts = torch.cat([xyz, pts[:, 3:] @ R.T], dim=1) if self.in_dim == 6 else xyz
            return pts

    train_ds = _View(full_ds, train_idx, aug=augment)
    val_ds   = _View(full_ds, val_idx,   aug=False)

    # Data already in RAM — num_workers=0 avoids multiprocess tensor pickling overhead
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    encoder = EncoderPointnet(
        latent_dim=args.latent_dim, in_dim=args.in_dim, global_dim=args.global_dim
    )
    decoder = (
        DecoderMLP(latent_dim=args.latent_dim, num_points=args.num_points)
        if args.decoder_type == "mlp"
        else DecoderFolding(latent_dim=args.latent_dim, num_points=args.num_points)
    )
    model = VaePointnet(encoder, decoder).to(device)

    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Output dirs ───────────────────────────────────────────────────────
    out_dir = Path(args.ckpt_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"

    tb_dir = Path(args.tb_dir) / args.exp_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {args.tb_dir}")

    start_epoch = 1
    best_val_cd = float("inf")

    # ── Optional resume ───────────────────────────────────────────────────
    last_ckpt = out_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_val_cd = ck.get("best_val_cd", float("inf"))
        print(f"Resumed from epoch {ck['epoch']}  (best val CD: {best_val_cd:.5f})")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "beta", "train_loss", "train_cd", "train_kl",
                "val_loss", "val_cd", "val_kl"
            ])

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        beta_cur = kl_weight(epoch, args.beta, args.kl_anneal_epochs)

        tr_loss, tr_cd, tr_kl = run_epoch(
            model, train_loader, optimizer,
            beta_cur, args.free_bits, args.tnet_reg, device, train=True,
            recon_loss=args.recon_loss, emd_weight=args.emd_weight,
            emd_iters=args.emd_iters, emd_n_subsample=args.emd_n_subsample,
        )
        vl_loss, vl_cd, vl_kl = run_epoch(
            model, val_loader, optimizer,
            beta_cur, args.free_bits, args.tnet_reg, device, train=False,
            recon_loss=args.recon_loss, emd_weight=args.emd_weight,
            emd_iters=args.emd_iters, emd_n_subsample=args.emd_n_subsample,
        )
        scheduler.step()

        print(
            f"[{epoch:03d}/{args.epochs}]  β={beta_cur:.5f} | "
            f"train  loss={tr_loss:.5f}  CD={tr_cd:.5f}  KL={tr_kl:.3f} | "
            f"val    loss={vl_loss:.5f}  CD={vl_cd:.5f}  KL={vl_kl:.3f}"
        )

        # ── TensorBoard scalars (every epoch) ─────────────────────────────
        writer.add_scalars("Loss/total", {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Loss/CD",    {"train": tr_cd,   "val": vl_cd},   epoch)
        writer.add_scalars("Loss/KL",    {"train": tr_kl,   "val": vl_kl},   epoch)
        writer.add_scalar("Params/beta", beta_cur,                            epoch)
        writer.add_scalar("Params/lr",   optimizer.param_groups[0]["lr"],     epoch)

        # ── post_beta scalars: same metrics but only after annealing ends ──
        # Gives a clean x-axis starting at 0 that shows only the regularised
        # training phase — easier to compare experiments with different anneal
        # lengths since the curves are always aligned at the same "post-anneal epoch".
        annealing_done = (epoch >= max(args.kl_anneal_epochs, 1))
        if annealing_done:
            post_epoch = epoch - max(args.kl_anneal_epochs, 1)
            writer.add_scalars("post_beta/Loss_total", {"train": tr_loss, "val": vl_loss}, post_epoch)
            writer.add_scalars("post_beta/Loss_CD",    {"train": tr_cd,   "val": vl_cd},   post_epoch)
            writer.add_scalars("post_beta/Loss_KL",    {"train": tr_kl,   "val": vl_kl},   post_epoch)

        # ── TensorBoard geometry + latent (every vis_every epochs) ────────
        if epoch % args.vis_every == 0 or epoch == 1:
            log_visuals(writer, model, val_loader, device, epoch, n_show=args.n_show)

        # ── CSV ───────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, beta_cur, tr_loss, tr_cd, tr_kl, vl_loss, vl_cd, vl_kl
            ])

        # ── Checkpoints ───────────────────────────────────────────────────
        # Only consider this epoch for best.pt once the KL annealing is done.
        # During annealing beta ≈ 0, so the model acts like an autoencoder and
        # gets artificially low CD — a checkpoint from that period would have a
        # poorly regularised latent space, useless for downstream diffusion.
        if annealing_done and vl_cd < best_val_cd:
            best_val_cd = vl_cd
            torch.save({
                "epoch": epoch, "val_cd": vl_cd,
                "model": model.state_dict(),
                "args":  vars(args),
            }, out_dir / "best.pt")
            writer.add_scalar("Val/best_CD", best_val_cd, epoch)

        torch.save({
            "epoch":       epoch,
            "val_cd":      vl_cd,
            "best_val_cd": best_val_cd,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
        }, out_dir / "last.pt")

    writer.close()
    print(f"\nDone.  Best val CD = {best_val_cd:.5f}  |  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
