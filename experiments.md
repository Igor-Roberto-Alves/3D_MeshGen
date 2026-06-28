# Experiment Setups

## How to run

Two phases per experiment — VAE first, then diffusion on the frozen VAE.
The diffusion script reads `latent_dim` and `style_dim` from the checkpoint automatically.

---

## Phase 1 — VAE experiments

### A. Sanity check (single class, fast)

Goal: confirm the pipeline runs end-to-end before investing in full training.
Use a single ShapeNet class (e.g. chairs `03001627`) or whatever your `point_clouds/` folder has.

```bash
python train_vae.py \
  --latent_dim 3 \
  --style_dim 128 \
  --epochs 150 \
  --batch_size 16 \
  --lr 3e-4 \
  --beta_start 1e-7 \
  --beta_end 1e-3 \
  --beta_epochs 50 \
  --recon_loss chamfer \
  --normal_weight 0.0 \
  --pos_noise_std 0.05 \
  --warmup_epochs 10 \
  --log_dir logs_vae_A
```

Expected by epoch 50: `val_recon < 1e-3`, `val_kl_style > 0.1`.
If `val_recon` is stuck above 5e-3 by epoch 30, the anchor skip is broken.

---

### B. Balanced — all classes, paper-close

```bash
python train_vae.py \
  --latent_dim 3 \
  --style_dim 256 \
  --epochs 300 \
  --batch_size 16 \
  --lr 3e-4 \
  --beta_start 1e-7 \
  --beta_end 5e-4 \
  --beta_epochs 150 \
  --recon_loss both \
  --emd_weight 0.2 \
  --normal_weight 0.1 \
  --pos_noise_std 0.05 \
  --warmup_epochs 15 \
  --log_dir logs_vae_B
```

Target: `val_recon < 5e-4`, `val_f_score > 0.70` by epoch 200.

---

### C. High quality

```bash
python train_vae.py \
  --latent_dim 6 \
  --style_dim 512 \
  --epochs 500 \
  --batch_size 8 \
  --lr 1e-4 \
  --beta_start 1e-7 \
  --beta_end 3e-4 \
  --beta_epochs 200 \
  --recon_loss both \
  --emd_weight 0.3 \
  --normal_weight 0.15 \
  --pos_noise_std 0.08 \
  --warmup_epochs 20 \
  --grad_clip 0.5 \
  --log_dir logs_vae_C
```

Target: `val_recon < 2e-4`, `val_f_score > 0.85`.

---

## Phase 2 — Diffusion experiments

Run after the VAE checkpoint is ready. Each points to its VAE checkpoint.

### D. Sanity diffusion (after VAE-A)

```bash
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --T 1000 \
  --style_hidden 256 \
  --style_layers 4 \
  --point_hidden 128 \
  --point_layers 6 \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-4 \
  --guidance 3.0 \
  --cfg_dropout 0.1 \
  --gen_metrics_every 25 \
  --log_dir logs_diff_D
```

---

### E. Balanced diffusion (after VAE-B)

```bash
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --T 1000 \
  --style_hidden 512 \
  --style_layers 6 \
  --point_hidden 256 \
  --point_layers 8 \
  --epochs 300 \
  --batch_size 16 \
  --lr 1e-4 \
  --guidance 3.0 \
  --cfg_dropout 0.1 \
  --gen_metrics_every 25 \
  --log_dir logs_diff_E
```

Target: `gen_nna ∈ [0.50, 0.60]`, `gen_cov > 0.40`, `gen_mmd < 1e-3`.

---

### F. Full quality diffusion (after VAE-C)

```bash
python train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \
  --T 2000 \
  --style_hidden 1024 \
  --style_layers 8 \
  --point_hidden 512 \
  --point_layers 10 \
  --epochs 500 \
  --batch_size 8 \
  --lr 5e-5 \
  --guidance 4.0 \
  --cfg_dropout 0.12 \
  --gen_metrics_every 50 \
  --log_dir logs_diff_F
```

Target: `gen_nna → 0.50`, `gen_cov > 0.50`, `gen_mmd < 5e-4`.

---

## Phase 3 — Ablation experiments

These tell you *which improvements in your architecture actually matter*.
All ablations use Setup B as the baseline (VAE-B + Diffusion-E).

### ABL-1: Position noise ablation

Tests whether `pos_noise_std` (the position shortcut break) matters.

| Run | `pos_noise_std` | Expected effect |
|-----|----------------|-----------------|
| B (baseline) | 0.05 | Decoder learns to use z_g for shape correction |
| ABL-1a | 0.0 | Decoder may rely too heavily on z_l positions → worse generation |
| ABL-1b | 0.20 | Too much noise → worse reconstruction → higher val_recon |

```bash
# ABL-1a — no noise (shortcut enabled)
python train_vae.py --pos_noise_std 0.0 --log_dir logs_vae_abl1a [same as B]

# ABL-1b — high noise
python train_vae.py --pos_noise_std 0.20 --log_dir logs_vae_abl1b [same as B]
```

Key metric to compare: `gen_nna` after diffusion. If ABL-1a gives worse `gen_nna`, position noise is helping.

---

### ABL-2: Normal loss ablation

Tests whether the normal prediction head helps geometry.

| Run | `normal_weight` | Expected effect |
|-----|----------------|-----------------|
| B (baseline) | 0.1 | Mild surface regularisation |
| ABL-2a | 0.0 | No normal constraint — possibly smoother chamfer |
| ABL-2b | 0.5 | Strong normal constraint — may hurt chamfer, improve surface quality |

```bash
python train_vae.py --normal_weight 0.0 --log_dir logs_vae_abl2a [same as B]
python train_vae.py --normal_weight 0.5 --log_dir logs_vae_abl2b [same as B]
```

Key metrics: `val_recon` (chamfer) and qualitative normal maps in TensorBoard.

---

### ABL-3: Latent dim ablation

Tests how much per-point feature capacity the model needs.

| Run | `latent_dim` | z_l per point | Expected effect |
|-----|-------------|--------------|-----------------|
| ABL-3a | 1 | (3+1)=4 | Underfitting, poor diversity |
| ABL-3b | 3 (LION paper) | (3+3)=6 | Balanced |
| B (baseline) | 3 | 6 | Same as paper |
| ABL-3c | 6 | (3+6)=9 | More capacity, check latent/active_units_local |
| ABL-3d | 8 | (3+8)=11 | High capacity |

```bash
python train_vae.py --latent_dim 1 --log_dir logs_vae_abl3a [same as B]
python train_vae.py --latent_dim 6 --log_dir logs_vae_abl3b [same as B]
python train_vae.py --latent_dim 8 --log_dir logs_vae_abl3c [same as B]
```

Watch `latent/active_units_local` in TensorBoard. If active units == latent_dim for ABL-3d, increase further. If active_units << latent_dim, you can reduce.

---

### ABL-4: CFG guidance scale sweep

Run on the same trained diffusion checkpoint (no retraining needed). Uses the `generate.py` script or the built-in generation inside `train_diffusion.py`.

| Guidance | Expected |
|----------|----------|
| 1.0 | No guidance, maximum diversity, lower quality |
| 2.0 | Mild guidance |
| 3.0 (default) | Good balance |
| 5.0 | High fidelity, low diversity |
| 7.0 | Over-sharpened, mode collapse risk |

To measure: run `_eval_generation` at different guidance values and compare `gen_nna`, `gen_cov`, `gen_mmd`.

---

### ABL-5: Reconstruction loss comparison

| Run | `recon_loss` | `emd_weight` | Expected |
|-----|-------------|-------------|---------|
| ABL-5a | chamfer | — | Fastest, slightly rougher surfaces |
| ABL-5b | emd | — | Slowest, best point uniformity |
| B (baseline) | both | 0.2 | Balanced |
| ABL-5c | both | 0.4 | More EMD weight |

```bash
python train_vae.py --recon_loss chamfer --log_dir logs_vae_abl5a [same as B]
python train_vae.py --recon_loss emd    --log_dir logs_vae_abl5b [same as B]
python train_vae.py --recon_loss both --emd_weight 0.4 --log_dir logs_vae_abl5c [same as B]
```

---

### ABL-6: KL annealing speed

Tests how fast beta grows from near-zero to `beta_end`.

| Run | `beta_epochs` | Effect |
|-----|--------------|--------|
| ABL-6a | 50 | Fast annealing — risk of early KL collapse |
| B (baseline) | 150 | Slow, stable |
| ABL-6b | 250 | Very slow — may underfit KL |

```bash
python train_vae.py --beta_epochs 50  --log_dir logs_vae_abl6a [same as B]
python train_vae.py --beta_epochs 250 --log_dir logs_vae_abl6b [same as B]
```

Watch `val_kl_style`: should reach 0.5–5.0. If it collapses to 0 → beta_epochs too short.

---

## Reading the metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `val_recon` | Chamfer / EMD distance to GT | Lower = better |
| `val_f_score` | % predicted points within 1cm of GT | > 0.70 good, > 0.85 excellent |
| `val_kl_points` | KL of local latent (feature part) | 0.01–0.5 |
| `val_kl_style` | KL of global latent | 0.5–5.0 |
| `latent/active_units_local` | How many z_l dims the encoder uses | Should equal latent_dim |
| `latent/active_units_global` | How many z_g dims the encoder uses | > 50% of style_dim |
| `gen_mmd` | Mean dist from real to closest generated (quality) | Lower = better |
| `gen_cov` | Fraction of real clouds matched (diversity) | Higher = better |
| `gen_nna` | How indistinguishable gen and real are | Closer to 0.5 = better |

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `val_recon` stuck > 5e-3 from epoch 0 | Anchor skip broken | Check `z_l[..,:3] = xyz + 0.01*delta` in Vae.forward |
| `val_kl_style` collapses to 0 early | `beta_end` too high or `beta_epochs` too short | Lower `beta_end` to 1e-4, extend `beta_epochs` |
| `active_units_local = 0` | All local KL clamped to free_bits immediately | Lower `beta_feat_pts` (try 1e-4) |
| `val_recon` oscillates without decreasing | LR too high or grad_clip too loose | Lower LR to 1e-4, `grad_clip=0.5` |
| `gen_nna = 1.0` | Generated clouds are random noise | More diffusion epochs; check VAE quality first |
| `gen_cov < 0.1` | Diffusion mode collapse | Lower `guidance`, increase `cfg_dropout` to 0.15 |
| `gen_nna ≈ 0.5` but `gen_cov < 0.3` | Good quality, no diversity | Increase `cfg_dropout` to 0.15–0.20 |
| Loss NaN after epoch 1 | AMP overflow | Add `--amp false` to diagnose; check logvar clamp |

---

## Recommended run order

1. **Setup A + D** — sanity check (~4h on a single GPU). If val_recon < 1e-3 by epoch 50, proceed.
2. **Setup B + E** — main result. (~2–3 days). This is your paper-quality baseline.
3. **ABL-1 and ABL-3** — highest signal ablations (position noise and latent dim).
4. **ABL-4 guidance sweep** — free (no retraining), done in ~1h with existing checkpoint.
5. **Setup C + F** — only if results from B are promising and you have GPU budget.
