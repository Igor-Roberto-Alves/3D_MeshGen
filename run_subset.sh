#!/bin/bash
# Treino sequencial para subconjunto: airplane + chair + car
# VAE ~3.7h (65 epochs) + Difusão ~1.6h (80 epochs) = ~5.3h total
# Inicie com: bash run_subset.sh | tee run_subset.log

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  LION subset: airplane + chair + car"
echo "  Início: $(date)"
echo "=========================================="

# ----------------------------------------------------------
# STAGE 1: VAE
# Params alinhados com LION:
#  - latent_dim=8  (LION usa 256; 3 era pequeno demais)
#  - lr=1e-4       (LION: learning_rate_vae=1e-4; 5e-4 causava NaN)
#  - beta_pos_pts=1.0, beta_feat_pts=1.0  (LION: weight_kl_pt=weight_kl_feat=1.0)
#  - beta_end=0.1  (peso final uniforme para todos os termos KL)
#  - amp=false     (float16 + logvar grande = overflow; sem AMP é estável)
# ----------------------------------------------------------
echo ""
echo "[$(date +%H:%M)] >>> STAGE 1: VAE (65 epochs)"
echo ""

python train_vae.py \
  --ckpt_dir        checkpoints_subset \
  --log_dir         logs_subset_vae \
  --latent_dim      8 \
  --style_dim       128 \
  --in_channels     6 \
  --epochs          65 \
  --batch_size      16 \
  --lr              1e-4 \
  --weight_decay    1e-4 \
  --warmup_epochs   10 \
  --grad_clip       1.0 \
  --beta_start      1e-7 \
  --beta_end        0.1 \
  --beta_epochs     30 \
  --beta_style      1.0 \
  --beta_pos_pts    1.0 \
  --beta_feat_pts   1.0 \
  --pos_noise_std   0.05 \
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
# Params alinhados com LION:
#  - LinearSchedule (LION usa linear, não cosine)
#  - amp=false (mais seguro; latentes padronizados reduzem risco mas não eliminam)
#  - point_dim inferido do VAE (3 + latent_dim = 3 + 8 = 11)
# ----------------------------------------------------------
echo "[$(date +%H:%M)] >>> STAGE 2: Difusão (80 epochs)"
echo ""

python train_diffusion.py \
  --vae_ckpt         checkpoints_subset/best.pt \
  --ckpt_dir         checkpoints_subset_diff \
  --log_dir          logs_subset_diff \
  --T                1000 \
  --num_classes      55 \
  --style_hidden     256 \
  --style_layers     4 \
  --cfg_dropout      0.1 \
  --guidance         3.0 \
  --point_hidden     128 \
  --point_layers     6 \
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
echo ""
echo "  Para gerar e avaliar:"
echo "    python generate.py --ckpt checkpoints_subset_diff/best.pt"
echo "=========================================="
