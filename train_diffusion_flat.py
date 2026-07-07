"""
train_diffusion_flat.py
-----------------------
Class-conditional latent diffusion (1-D U-Net) on top of the frozen flat
VAE (VaePointnet, checkpoint saved by train_vaepointnet.py).

Pipeline
  1. Load the trained VAE checkpoint; rebuild the EXACT same train/val
     split used during VAE training (same seed / val_frac / sorted file
     list), so the diffusion never sees VAE-validation shapes.
  2. Encode every cloud once with the frozen encoder → (mu, std) latents
     of size latent_dim (256 or 512 depending on the VAE config).
  3. Normalise latents per-dimension (train statistics) and train a
     class-conditional UNet1D to predict the DDPM noise (eps-MSE),
     with classifier-free guidance dropout, EMA weights, warmup+cosine
     LR and optional Min-SNR-gamma loss weighting.
  4. Periodically sample per class (DDIM + CFG), decode with the frozen
     VAE decoder and log MMD-CD / COV-CD / 1-NNA-CD against the real
     validation clouds of that class.

Classes (from the PLY filename prefix in point_clouds/):
  0 = airplane (02691156)     1 = car (02958343)

Quick start
-----------
  python train_diffusion_flat.py --exp_name exp0
  python train_diffusion_flat.py --exp_name exp1_deep --ch_mult 1,2,4,4
  python train_diffusion_flat.py --exp_name exp2_postz --posterior_sample
  python train_diffusion_flat.py --exp_name exp0 --resume

TensorBoard:  tensorboard --logdir logs_diff_flat
"""

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from EncoderPointnet import EncoderPointnet
from DecoderPointnet import DecoderMLP, DecoderFolding
from Vaepointnet import VaePointnet
from train_vaepointnet import _load_ply, CATEGORY_IDS
from train_vae_flat import _side_by_side
from src.dataset import Ds_point_sampled_already
from src.vae_flat import VaeFlat
from src.Vae import normalize_pc
from src.Diffusion import CosineSchedule
from src.UnetDiffusion import UNet1D, ddim_sample
from src.metric import generative_metrics


CLASS_IDS   = {"02691156": 0, "02958343": 1}
CLASS_NAMES = {0: "airplane", 1: "car"}
NUM_CLASSES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Data / VAE helpers (also imported by final_metrics.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def list_files_with_labels(data_root: str, category: str = "all"):
    """Sorted PLY list + integer class labels. Same ordering/filter as
    train_vaepointnet.PointCloudDataset, so index-based splits match."""
    cat_id = CATEGORY_IDS.get(category)
    files, labels = [], []
    for f in sorted(Path(data_root).glob("*.ply")):
        if cat_id is None or f.name.startswith(cat_id):
            files.append(f)
            labels.append(CLASS_IDS[f.name.split("_")[0]])
    if not files:
        raise RuntimeError(f"No PLY files found in '{data_root}' (category={category})")
    return files, labels


def make_split(n: int, seed: int, val_frac: float):
    """Exact replica of the split in train_vaepointnet.main()."""
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(n * val_frac))
    all_idx = torch.randperm(n, generator=g).tolist()
    return all_idx[: n - n_val], all_idx[n - n_val:]


def list_files_flat(data_root: str):
    """PLY list in the SAME (os.walk) order as Ds_point_sampled_already, so
    index-based splits match the ones used in train_vae_flat.main()."""
    ds = Ds_point_sampled_already(root=data_root, augment=False)
    files  = [Path(f) for f in ds.files]
    labels = [CLASS_IDS[f.name.split("_")[0]] for f in files]
    return files, labels


def make_split_flat(n: int, seed: int, val_frac: float):
    """Exact replica of the split in train_vae_flat.main() (val = FIRST slice
    of the permutation, unlike make_split above which takes the last)."""
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(n * val_frac))
    all_idx = torch.randperm(n, generator=g).tolist()
    return all_idx[n_val:], all_idx[:n_val]


def load_clouds(files, in_dim: int = 3) -> torch.Tensor:
    """Load + normalise (unit sphere) all clouds → [M, N, in_dim] float32."""
    return torch.stack([_load_ply(f, in_dim) for f in files])


def build_vae(vae_args: dict, device) -> VaePointnet:
    encoder = EncoderPointnet(
        latent_dim=vae_args["latent_dim"],
        in_dim=vae_args["in_dim"],
        global_dim=vae_args["global_dim"],
    )
    decoder_cls = DecoderMLP if vae_args["decoder_type"] == "mlp" else DecoderFolding
    decoder = decoder_cls(latent_dim=vae_args["latent_dim"],
                          num_points=vae_args["num_points"])
    return VaePointnet(encoder, decoder).to(device)


def build_vae_flat(cfg: dict, decoder_norm: str, device) -> VaeFlat:
    return VaeFlat(
        latent_dim=cfg["latent_dim"], in_channels=cfg["in_channels"],
        n_latent=cfg["n_latent"], n_points=cfg["n_points"],
        seed_dim=cfg["seed_dim"],
        hidden_dim=cfg["decoder_hidden_dim"], n_stages=cfg["decoder_stages"],
        fold_dim=cfg["decoder_fold_dim"], fold_hidden=cfg["decoder_fold_hidden"],
        resolution=cfg["decoder_resolution"],
        encoder_hidden_dim=cfg["encoder_hidden_dim"],
        encoder_out_dim=cfg["encoder_out_dim"],
        encoder_stages=cfg["encoder_stages"],
        encoder_resolution=cfg["encoder_resolution"],
        decoder_norm=decoder_norm,
    ).to(device)


def load_vae(ckpt_path: str, device):
    """Load a frozen VAE + normalised training args from a best.pt/last.pt.

    Supports both checkpoint layouts:
      {"args": ...}   — VaePointnet, saved by train_vaepointnet.py
      {"config": ...} — VaeFlat,     saved by train_vae_flat.py
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ck:                                     # VaeFlat
        cfg, state = ck["config"], ck["model"]
        # Checkpoints trained before the BatchNorm→GroupNorm switch in
        # src.Decoder.SharedMLP carry BN running stats; rebuild accordingly.
        norm = "batch" if "decoder.feat_proj.1.running_mean" in state else "group"
        vae = build_vae_flat(cfg, norm, device)
        vae.load_state_dict(state)
        vae_args = {
            "arch":       "flat",
            "latent_dim": cfg["latent_dim"],
            "in_dim":     cfg["in_channels"],
            "data_root":  cfg["data_root"],
            "category":   "all",          # Ds_point_sampled_already uses every PLY
            "seed":       cfg["seed"],
            "val_frac":   cfg["val_split"],
        }
    else:                                                  # VaePointnet
        vae_args = dict(ck["args"], arch="pointnet")
        vae = build_vae(vae_args, device)
        vae.load_state_dict(ck["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, vae_args


@torch.no_grad()
def encode_latents(vae, clouds: torch.Tensor, device, batch: int = 64):
    """Frozen-encoder pass → posterior (mu, std), both [M, latent_dim] on CPU."""
    vae.eval()
    flat = isinstance(vae, VaeFlat)
    mus, stds = [], []
    for s in range(0, clouds.shape[0], batch):
        x = clouds[s:s + batch].to(device)
        if flat:
            mu, logvar = vae.encoder(x)                       # [b, N, C], já normalize_pc
        else:
            mu, logvar, _ = vae.encoder(x.permute(0, 2, 1))   # [b, in_dim, N]
        mus.append(mu.cpu())
        stds.append((0.5 * logvar.clamp(-10, 10)).exp().cpu())
    return torch.cat(mus), torch.cat(stds)


def subsample_points(clouds: torch.Tensor, n: int) -> torch.Tensor:
    """Random n-point subset per cloud (for the O(N^2) set metrics)."""
    if clouds.shape[1] <= n:
        return clouds
    idx = torch.rand(clouds.shape[0], clouds.shape[1],
                     device=clouds.device).argsort(dim=1)[:, :n]
    return clouds.gather(1, idx.unsqueeze(-1).expand(-1, -1, clouds.shape[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Sampling / decoding (also used by the notebook)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_latents(model: UNet1D, schedule: CosineSchedule, n: int,
                   class_id: int, latent_mean: torch.Tensor,
                   latent_std: torch.Tensor, device,
                   guidance: float = 2.0, ddim_steps: int = 200,
                   batch: int = 64) -> torch.Tensor:
    """Sample n DE-normalised latents of one class → [n, latent_dim] on device."""
    model.eval()
    D = model.latent_dim
    out = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        cond = torch.full((b,), class_id, device=device, dtype=torch.long)
        z = ddim_sample(schedule, model, (D,), cond,
                        uncond=model.uncond(b, device),
                        guidance=guidance, steps=ddim_steps)
        out.append(z)
    z = torch.cat(out)
    return z * latent_std.to(device) + latent_mean.to(device)


@torch.no_grad()
def decode_latents(vae: VaePointnet, z: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """Frozen-decoder pass → [n, num_points, 3] on z's device."""
    vae.eval()
    return torch.cat([vae.decoder(z[s:s + batch]) for s in range(0, z.shape[0], batch)])


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ─────────────────────────────────────────────────────────────────────────────
# Generative evaluation (sample → decode → set metrics vs. real val clouds)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gen_eval(model, schedule, vae, latent_mean, latent_std,
             val_clouds, val_labels, device, n_gen=64, guidance=2.0,
             ddim_steps=200, metric_points=1024, metric_chunk=2,
             writer=None, epoch=None, n_vis=4):
    results = {}
    for c in range(NUM_CLASSES):
        refs = val_clouds[val_labels == c][..., :3]
        n = min(n_gen, refs.shape[0])
        ridx = torch.randperm(refs.shape[0])[:n]
        ref  = subsample_points(refs[ridx].to(device), metric_points)

        z        = sample_latents(model, schedule, n, c, latent_mean, latent_std,
                                  device, guidance=guidance, ddim_steps=ddim_steps)
        gen_full = decode_latents(vae, z)                       # [n, N, 3]
        gen      = subsample_points(gen_full, metric_points)

        m = generative_metrics(gen, ref, chunk=metric_chunk)
        results[CLASS_NAMES[c]] = {k: float(v) for k, v in m.items()}

        # Amostras geradas no TensorBoard (real verde | gerada azul) —
        # mesmo padrao visual do log_reconstructions do train_vae_flat.
        if writer is not None:
            for i in range(min(n_vis, gen_full.shape[0])):
                ref_v = refs[ridx[i]].float().cpu()
                gen_v = gen_full[i].float().cpu()
                ref_c = torch.tensor([[0, 220, 0]],  dtype=torch.uint8).expand(ref_v.shape[0], -1)
                gen_c = torch.tensor([[0, 80, 220]], dtype=torch.uint8).expand(gen_v.shape[0], -1)
                verts, clrs = _side_by_side([ref_v, gen_v], [ref_c, gen_c])
                writer.add_mesh(f"gen_{CLASS_NAMES[c]}/sample_{i}",
                                vertices=verts, colors=clrs, global_step=epoch)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── VAE / data ────────────────────────────────────────────────────────
    p.add_argument("--vae_ckpt",  default="ckpt_pointvae/exp0_baseline/best.pt")
    p.add_argument("--data_root", default=None,
                   help="Override PLY dir (default: the one saved in the VAE ckpt)")
    p.add_argument("--posterior_sample", action="store_true",
                   help="Train on z = mu + std*eps (fresh noise each step) "
                        "instead of the posterior mean mu")

    # ── U-Net ─────────────────────────────────────────────────────────────
    p.add_argument("--base_ch",      type=int,   default=64)
    p.add_argument("--ch_mult",      default="1,2,4",
                   help="Comma-separated channel multipliers per U-Net level")
    p.add_argument("--n_res_blocks", type=int,   default=2)
    p.add_argument("--attn_levels",  default="1,2",
                   help="Comma-separated levels that get self-attention ('' = none)")
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--time_dim",     type=int,   default=128)
    p.add_argument("--cfg_dropout",  type=float, default=0.1)

    # ── Diffusion / training ──────────────────────────────────────────────
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--epochs",        type=int,   default=400)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--min_snr_gamma", type=float, default=5.0,
                   help="Min-SNR-gamma loss weighting (<=0 disables)")
    p.add_argument("--ema_decay",     type=float, default=0.999)

    # ── Generative eval during training ──────────────────────────────────
    p.add_argument("--gen_every",     type=int,   default=50,
                   help="Sample+decode+set-metrics every N epochs (0 = off)")
    p.add_argument("--n_gen",         type=int,   default=64)
    p.add_argument("--guidance",      type=float, default=2.0)
    p.add_argument("--ddim_steps",    type=int,   default=200)
    p.add_argument("--metric_points", type=int,   default=1024)
    p.add_argument("--metric_chunk",  type=int,   default=2)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--exp_name", default="exp0")
    p.add_argument("--ckpt_dir", default="ckpt_diff_flat")
    p.add_argument("--tb_dir",   default="logs_diff_flat")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--seed",     type=int, default=42)

    args = p.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Frozen VAE + exact same split as its training ─────────────────────
    vae, vae_args = load_vae(args.vae_ckpt, device)
    data_root  = args.data_root or vae_args["data_root"]
    latent_dim = vae_args["latent_dim"]
    print(f"VAE: {args.vae_ckpt}  |  latent_dim={latent_dim}  "
          f"|  category={vae_args['category']}")

    if vae_args["arch"] == "flat":
        files, labels = list_files_flat(data_root)
        train_idx, val_idx = make_split_flat(len(files), vae_args["seed"],
                                             vae_args["val_frac"])
    else:
        files, labels = list_files_with_labels(data_root, vae_args["category"])
        train_idx, val_idx = make_split(len(files), vae_args["seed"],
                                        vae_args["val_frac"])
    labels_t = torch.tensor(labels, dtype=torch.long)
    print(f"Files: {len(files)}  |  train={len(train_idx)}  val={len(val_idx)}")

    print("Loading clouds into RAM ...", flush=True)
    clouds = load_clouds(files, vae_args["in_dim"])            # [M, N, in_dim]
    if vae_args["arch"] == "flat":
        # VaeFlat normalises inside forward(); apply it once here so the
        # encoder input AND the metric reference clouds live in the same
        # space as the decoder output.
        clouds = normalize_pc(clouds)

    print("Encoding latents with the frozen VAE encoder ...", flush=True)
    mu, std = encode_latents(vae, clouds, device)              # [M, D] each

    tr = torch.tensor(train_idx)
    vl = torch.tensor(val_idx)

    # Per-dim normalisation from TRAIN statistics only
    latent_mean = mu[tr].mean(dim=0)
    latent_std  = mu[tr].std(dim=0).clamp(min=1e-3)
    print(f"Latents: mu-std per dim in [{latent_std.min():.4f}, {latent_std.max():.4f}]  "
          f"|  active dims (std>0.1): {(mu[tr].std(dim=0) > 0.1).sum().item()}/{latent_dim}")

    train_loader = DataLoader(
        TensorDataset(mu[tr], std[tr], labels_t[tr]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(mu[vl], std[vl], labels_t[vl]),
        batch_size=args.batch_size, shuffle=False,
    )
    val_clouds = clouds[vl]           # real clouds for the set metrics
    val_labels = labels_t[vl]

    # ── Model / schedule / optimiser ──────────────────────────────────────
    model_kwargs = dict(
        latent_dim=latent_dim,
        num_classes=NUM_CLASSES,
        base_ch=args.base_ch,
        ch_mult=tuple(int(x) for x in args.ch_mult.split(",")),
        n_res_blocks=args.n_res_blocks,
        attn_levels=tuple(int(x) for x in args.attn_levels.split(",") if x != ""),
        n_heads=args.n_heads,
        time_dim=args.time_dim,
        cfg_dropout=args.cfg_dropout,
    )
    model = UNet1D(**model_kwargs).to(device)
    print(f"UNet1D parameters: {sum(q.numel() for q in model.parameters()):,}")

    schedule = CosineSchedule(T=args.T).to(device)
    ema       = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    ema_model = copy.deepcopy(model).eval() if ema is not None else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(epoch):   # linear warmup → cosine decay (per epoch)
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Min-SNR-gamma weights per timestep (eps-prediction): min(SNR, g)/SNR
    snr = schedule.acp / (1 - schedule.acp)
    snr_w = (snr.clamp(max=args.min_snr_gamma) / snr) if args.min_snr_gamma > 0 \
        else torch.ones_like(snr)

    # ── Output dirs ───────────────────────────────────────────────────────
    out_dir = Path(args.ckpt_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    writer = SummaryWriter(log_dir=str(Path(args.tb_dir) / args.exp_name))
    print(f"TensorBoard: tensorboard --logdir {args.tb_dir}")

    start_epoch, best_val = 1, float("inf")

    def save_ckpt(path, epoch, val_loss):
        torch.save({
            "epoch":        epoch,
            "val_loss":     val_loss,
            "best_val":     best_val,
            "model":        model.state_dict(),
            "ema":          ema.shadow if ema is not None else None,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "args":         vars(args),
            "model_kwargs": model_kwargs,
            "vae_ckpt":     args.vae_ckpt,
            "vae_args":     vae_args,
            "latent_mean":  latent_mean,
            "latent_std":   latent_std,
            "class_names":  CLASS_NAMES,
        }, path)

    last_ckpt = out_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ema is not None and ck.get("ema") is not None:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_epoch = ck["epoch"] + 1
        best_val    = ck.get("best_val", float("inf"))
        print(f"Resumed from epoch {ck['epoch']}  (best val loss: {best_val:.6f})")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "val_loss"]
                + [f"{n}_{m}" for n in CLASS_NAMES.values()
                   for m in ("mmd", "cov", "nna_1nn")]
            )

    mean_d = latent_mean.to(device)
    std_d  = latent_std.to(device)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # --- train -----------------------------------------------------
        model.train()
        tr_loss, n_b = 0.0, 0
        for mu_b, std_b, lab_b in train_loader:
            mu_b, std_b, lab_b = mu_b.to(device), std_b.to(device), lab_b.to(device)

            z = mu_b + std_b * torch.randn_like(std_b) if args.posterior_sample else mu_b
            z = (z - mean_d) / std_d

            t = torch.randint(0, args.T, (z.shape[0],), device=device)
            zt, noise = schedule.q_sample(z, t)
            eps = model(zt, t, lab_b)

            loss = ((eps - noise) ** 2).mean(dim=1)
            loss = (snr_w[t] * loss).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            tr_loss += loss.item()
            n_b += 1
        tr_loss /= max(n_b, 1)
        scheduler.step()

        # --- val (fixed t/noise seed → comparable across epochs) --------
        model.eval()
        g = torch.Generator().manual_seed(args.seed)
        vl_loss, n_b = 0.0, 0
        with torch.no_grad():
            for mu_b, _, lab_b in val_loader:
                z = ((mu_b.to(device) - mean_d) / std_d)
                t = torch.randint(0, args.T, (z.shape[0],), generator=g).to(device)
                noise = torch.randn(z.shape, generator=g).to(device)
                zt, _ = schedule.q_sample(z, t, noise)
                eps = model(zt, t, lab_b.to(device))
                vl_loss += ((eps - noise) ** 2).mean().item()
                n_b += 1
        vl_loss /= max(n_b, 1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{args.epochs}]  lr={lr_now:.2e} | "
              f"train eps-MSE={tr_loss:.6f} | val eps-MSE={vl_loss:.6f}")

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   vl_loss, epoch)
        writer.add_scalar("Params/lr",  lr_now,  epoch)

        # --- periodic generative eval (EMA weights) ---------------------
        gen_row = [""] * (3 * NUM_CLASSES)
        if args.gen_every > 0 and (epoch % args.gen_every == 0 or epoch == args.epochs):
            if ema is not None:
                ema.copy_to(ema_model)
            res = gen_eval(ema_model, schedule, vae, latent_mean, latent_std,
                           val_clouds, val_labels, device,
                           n_gen=args.n_gen, guidance=args.guidance,
                           ddim_steps=args.ddim_steps,
                           metric_points=args.metric_points,
                           metric_chunk=args.metric_chunk,
                           writer=writer, epoch=epoch)
            gen_row = []
            for c, name in CLASS_NAMES.items():
                m = res[name]
                gen_row += [m["mmd"], m["cov"], m["nna_1nn"]]
                for k, v in m.items():
                    writer.add_scalar(f"Gen/{name}_{k}", v, epoch)
                print(f"    gen[{name}]  MMD-CD={m['mmd']:.5f}  "
                      f"COV={m['cov']:.3f}  1-NNA={m['nna_1nn']:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr_now, tr_loss, vl_loss] + gen_row)

        # --- checkpoints -------------------------------------------------
        if vl_loss < best_val:
            best_val = vl_loss
            save_ckpt(out_dir / "best.pt", epoch, vl_loss)
        save_ckpt(last_ckpt, epoch, vl_loss)

    writer.close()
    print(f"\nDone.  Best val eps-MSE = {best_val:.6f}  |  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
