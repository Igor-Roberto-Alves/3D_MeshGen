FINAL_A:

python train_vae_up.py \
  --ckpt_dir  checkpoints_up_v2b \
  --log_dir   logs_up_v2b \
  --data_root point_clouds \
  --epochs 200 \
  --batch_size 8 \
  --free_bits 0.0 \
  --beta_end 0.5 \
  --beta_epochs 120 \
  --coarse_weight 0.5 \
  --recon_loss both \
  --emd_weight 0.5

Final B:

python train_vae_up.py \
  --ckpt_dir  checkpoints_up_v2 \
  --log_dir   logs_up_v2 \
  --data_root point_clouds \
  --epochs 200 \
  --batch_size 8 \
  --free_bits 0.0 \
  --beta_end 1.0 \
  --beta_epochs 100 \
  --coarse_weight 0.5 \
  --recon_loss both \
  --emd_weight 0.5