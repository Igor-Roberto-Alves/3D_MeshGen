#!/bin/bash
# Treino sequencial para subconjunto: airplane + chair + car
# Real_latent architecture: flat z_l ∈ ℝ^{latent_size}, sem skip de coordenadas
# VAE ~4h (65 epochs) + Difusão ~1.5h (80 epochs)
# Inicie com: bash run_subset.sh | tee run_subset.log

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  Real_latent subset: airplane + chair + car"
echo "  Início: $(date)"
echo "=========================================="

# ----------------------------------------------------------
# STAGE 1: VAE
#  - latent_size=1024  (flat global shape code; sem per-point)
#  - beta_shape=1.0, beta_style=1.0 (KL uniforme nos dois vetores)
#  - amp=false         (mais estável; sem AMP o VAE converge melhor)
# ----------------------------------------------------------
echo ""
echo "[$(date +%H:%M)] >>> STAGE 1: VAE (65 epochs)"
echo ""

python train_vae.py \
  --ckpt_dir        checkpoints_subset \
  --log_dir         logs_subset_vae \
  --latent_size     10000 \
  --style_dim       128 \
  --in_channels     6 \
  --num_points      2048 \
  --epochs          200 \
  --batch_size      16 \
  --lr              1e-4 \
  --weight_decay    1e-4 \
  --warmup_epochs   10 \
  --grad_clip       1.0 \
  --beta_start      1e-7 \
  --beta_end        0.1 \
  --beta_epochs     30 \
  --beta_style      1.0 \
  --beta_shape      1.0 \
  --recon_loss      chamfer \
  --val_split       0.1 \
  --save_every      1000 \
  --log_every       1 \
  --device          cuda \
  --amp             false

echo ""
echo "[$(date +%H:%M)] <<< VAE concluído. Melhor checkpoint: checkpoints_subset/best.pt"
echo ""

# ----------------------------------------------------------
# STAGE 2: Difusão
#  - FlatDenoiser para z_l (MLP DDPM, muito mais leve que LatentPointDenoiser)
#  - amp=false (consistente com VAE stage)
# ----------------------------------------------------------
echo "[$(date +%H:%M)] >>> STAGE 2: Difusão (80 epochs)"
echo ""

python train_diffusion.py \
  --vae_ckpt         checkpoints_subset/best.pt \
  --ckpt_dir         checkpoints_subset_diff \
  --log_dir          logs_subset_diff \
  --T                1000 \
  --num_classes      55 \
  --style_hidden     512 \
  --style_layers     6 \
  --cfg_dropout      0.1 \
  --guidance         3.0 \
  --shape_hidden     512 \
  --shape_layers     8 \
  --epochs           80 \
  --batch_size       16 \
  --lr               2e-4 \
  --weight_decay     1e-4 \
  --warmup_epochs    5 \
  --grad_clip        1.0 \
  --val_split        0.1 \
  --save_every       1000 \
  --log_every        1 \
  --device           cuda \
  --amp              false \
  --vis_every        10 \
  --vis_per_class    3 \
  --gen_metrics_every 20 \
  --gen_metrics_n    256

echo ""
echo "[$(date +%H:%M)] <<< Difusão concluída."
echo ""
echo "=========================================="
echo "  PRONTO. $(date)"
echo "  Resultados em: logs_subset_diff/"
echo "  Checkpoint:    checkpoints_subset_diff/best.pt"
echo "=========================================="
