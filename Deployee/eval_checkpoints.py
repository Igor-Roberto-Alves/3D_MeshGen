"""
eval_checkpoints.py
-------------------
Avalia os checkpoints salvos da difusao unconditional (ckpt_diff_flat_noclass)
com o MESMO protocolo do gen_eval do treino (EMA, 64 amostras, DDIM 200,
1024 pontos por nuvem, mesma seed) e grava Deployee/checkpoints_metrics.json.

O app.py usa esse JSON para resolver qual checkpoint atende cada criterio
(menor val loss, maior COV, menor MMD, 1-NNA mais perto de 0.5).

Rodar a partir da raiz do repo:
    .venv/bin/python Deployee/eval_checkpoints.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.Diffusion import CosineSchedule
from src.UnetDiffusion import UNet1D
from src.Vae import normalize_pc
from src.metric import generative_metrics
from train_diffusion_flat_noclass import (
    load_vae, list_files_flat, make_split_flat, load_clouds,
    sample_latents, decode_latents, subsample_points,
)

CKPT_DIR = ROOT / "ckpt_diff_flat_noclass" / "exp0_noclass"
OUT_PATH = Path(__file__).resolve().parent / "checkpoints_metrics.json"

N_GEN         = 64
DDIM_STEPS    = 200
METRIC_POINTS = 1024
METRIC_CHUNK  = 2
SEED          = 42


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {CKPT_DIR}")

    # VAE congelado + val split identico ao treino (info salva no proprio ckpt)
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    vae, vae_args = load_vae(str(ROOT / ck0["vae_ckpt"]), device)

    files, _ = list_files_flat(str(ROOT / vae_args["data_root"]))
    _, val_idx = make_split_flat(len(files), vae_args["seed"], vae_args["val_frac"])
    print(f"Val clouds: {len(val_idx)}")

    clouds = load_clouds([files[i] for i in val_idx], vae_args["in_dim"])
    refs = normalize_pc(clouds)[..., :3]

    results = {}
    for path in ckpts:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = UNet1D(**ck["model_kwargs"]).to(device)
        state = ck["ema"] if ck.get("ema") is not None else ck["model"]
        model.load_state_dict(state)
        model.eval()

        schedule = CosineSchedule(T=ck["args"]["T"]).to(device)

        torch.manual_seed(SEED)
        n = min(N_GEN, refs.shape[0])
        ridx = torch.randperm(refs.shape[0])[:n]
        ref = subsample_points(refs[ridx].to(device), METRIC_POINTS)

        z = sample_latents(model, schedule, n,
                           ck["latent_mean"], ck["latent_std"],
                           device, ddim_steps=DDIM_STEPS)
        gen = subsample_points(decode_latents(vae, z), METRIC_POINTS)

        m = generative_metrics(gen, ref, chunk=METRIC_CHUNK)
        results[path.stem] = {
            "file":     path.name,
            "epoch":    int(ck["epoch"]),
            "val_loss": float(ck["val_loss"]),
            "mmd":      float(m["mmd"]),
            "cov":      float(m["cov"]),
            "nna_1nn":  float(m["nna_1nn"]),
        }
        print(f"{path.name}: epoch={ck['epoch']}  val={ck['val_loss']:.5f}  "
              f"MMD={m['mmd']:.5f}  COV={m['cov']:.3f}  1-NNA={m['nna_1nn']:.3f}")

        n_dead = int((ck["latent_std"] == 1.0).sum())
        top = ck["latent_std"].argsort(descending=True)[:8].tolist()
        print(f"    latent_std: dead(=1.0)={n_dead}  top8 dims={top}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
