# Experiment Setups

Three setups ordered from "sanity check" to "full quality".
Run each as two phases: VAE first, then diffusion on the frozen VAE.

---

## Setup A — Sanity check (fast, single class)

Goal: confirm the pipeline runs end-to-end and reconstructions are good before wasting GPU time.
Use only one ShapeNet class (e.g. chairs: `03001627`).

**VAE**
```
python train_vae.py \
  --latent_dim 3 \
  --style_dim 128 \
  --epochs 150 \
  --batch_size 16 \
  --lr 3e-4 \
  --beta_start 0.0 \
  --beta_end 1e-3 \
  --beta_epochs 50 \
  --recon_loss chamfer \
  --emd_weight 0.0 \
  --warmup_epochs 10
```

**Diffusion**
```
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --vae_latent_dim 3 \
  --vae_style_dim 128 \
  --T 1000 \
  --style_hidden 256 \
  --style_layers 4 \
  --point_hidden 128 \
  --point_layers 6 \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-4 \
  --gen_metrics_every 25
```

**What to look for:**
- VAE val recon < 1e-3 by epoch 50 (sanity passes)
- Diffusion val losses decreasing and not diverging
- 1-NNA approaching 0.5 after epoch 100 (generated ≈ real)

---

## Setup B — Balanced (all classes, paper-close)

Goal: train on full ShapeNet, close to the LION paper config.

**VAE**
```
python train_vae.py \
  --latent_dim 3 \
  --style_dim 256 \
  --epochs 300 \
  --batch_size 16 \
  --lr 3e-4 \
  --beta_start 0.0 \
  --beta_end 1e-3 \
  --beta_epochs 100 \
  --recon_loss both \
  --emd_weight 0.2 \
  --warmup_epochs 15
```

**Diffusion**
```
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --vae_latent_dim 3 \
  --vae_style_dim 256 \
  --T 1000 \
  --style_hidden 512 \
  --style_layers 6 \
  --point_hidden 256 \
  --point_layers 8 \
  --epochs 300 \
  --batch_size 16 \
  --lr 1e-4 \
  --guidance 3.0 \
  --gen_metrics_every 25
```

**What to look for:**
- VAE: val recon < 5e-4, val f_score > 0.7 by epoch 200
- Diffusion: 1-NNA in [0.50, 0.60] (good), COV > 0.40, MMD < 1e-3

---

## Setup C — High quality (larger networks, cosine T=2000)

Goal: maximum quality. Requires more VRAM and time.

**VAE**
```
python train_vae.py \
  --latent_dim 3 \
  --style_dim 512 \
  --epochs 500 \
  --batch_size 8 \
  --lr 1e-4 \
  --beta_start 0.0 \
  --beta_end 5e-4 \
  --beta_epochs 150 \
  --recon_loss both \
  --emd_weight 0.3 \
  --warmup_epochs 20 \
  --grad_clip 0.5
```

**Diffusion**
```
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --vae_latent_dim 3 \
  --vae_style_dim 512 \
  --T 2000 \
  --style_hidden 1024 \
  --style_layers 8 \
  --point_hidden 512 \
  --point_layers 10 \
  --epochs 500 \
  --batch_size 8 \
  --lr 5e-5 \
  --guidance 4.0 \
  --gen_metrics_every 50
```

**What to look for:**
- VAE: val recon < 2e-4, val f_score > 0.85
- Diffusion: 1-NNA → 0.50 (perfect), COV > 0.50, MMD < 5e-4

---

## Reading the metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `val_recon` (VAE) | Reconstruction quality (Chamfer/EMD) | Lower = better |
| `val_f_score` (VAE) | % points within 1cm of GT | > 0.7 good, > 0.85 excellent |
| `val_kl_points` | KL of local latent — should be small but nonzero | 0.01–0.5 |
| `val_kl_style` | KL of global latent | 0.5–5.0 |
| `gen_mmd` | Distance from real to closest generated (quality) | Lower = better |
| `gen_cov` | Fraction of real shapes that have a generated match (diversity) | Higher = better |
| `gen_nna` | How indistinguishable gen and real are (both quality+diversity) | Closer to 0.5 = better |

### Failure modes to watch

| Symptom | Cause | Fix |
|---------|-------|-----|
| VAE recon stuck high from epoch 0 | Init broken — anchor skip not working | Check z_l = xyz + 0.01*mu_l |
| KL collapses to 0 immediately | beta_end too high, warmup too short | Lower beta_end or increase beta_epochs |
| gen_nna = 1.0 | Generated clouds are garbage (mode collapse) | Lower guidance, more diffusion epochs |
| gen_cov < 0.1 | Mode collapse in diffusion | Lower lambda_style, check CFG dropout |
| gen_nna = 0.5 but gen_cov low | Good quality but no diversity | Increase cfg_dropout to 0.15–0.2 |
