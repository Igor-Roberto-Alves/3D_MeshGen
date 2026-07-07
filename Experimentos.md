Exp: A

latent_dim: 3

beta_end: 1.0

beta_epochs: 100

recon_loss: chamfer

hipótese: baseline conservador

──────────────────────────────

Exp: B

latent_dim: 8

beta_end: 1.0

beta_epochs: 100

recon_loss: chamfer

hipótese: mais capacidade, mesma pressão

python3 train_vae.py --latent_dim 8 --beta_end 1 --beta_epochs 100 --recon_loss both --batch_size 8 --log_dir ExperimentB_logs --ckpt_dir ExperimentB_ckpt --save_every 100 --epochs 600

───────────────────────────────

Exp: C

latent_dim: 8

beta_end: 2.0

beta_epochs: 150

recon_loss: chamfer

hipótese: capacidade + mais tempo livre

───────────────────────────────

Exp: D

latent_dim: 8

beta_end: 1.0

beta_epochs: 100

recon_loss: both

hipótese: sinal mais rico

──────────────────────────────
Exp: E

latent_dim: 16

beta_end: 2.0

beta_epochs: 150

recon_loss: chamfer

hipótese: limite superior de capacidade