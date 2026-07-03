"""
overfit_point.py
----------------
Teste decisivo de sanidade do Stage 2 (LatentPointDenoiser / DiT).

Pega UM batch fixo, extrai os latentes limpos do VAE congelado e treina APENAS
o point_dn nesse batch por N steps. Com x0 fixo, o ruído ε é exatamente
determinado dado (xt, t), então um denoiser saudável MEMORIZA e leva a
loss_point para ~0.

Interpretação:
  - loss cai para ~0.02-0.05  -> arquitetura/otimização OK.
      => 0.36 no dataset cheio é PISO DE DADOS (latente do VAE pobre) -> mexer no VAE.
  - loss trava em ~0.36 até aqui -> há um GARGALO/BUG real no denoiser -> investigar o modelo.

Uso (mesmos args do train_diffusion.py que importam):
  python overfit_point.py --vae_ckpt checkpoints/best.pt --data_root point_clouds \
         --vae_type up --vae_latent_dim 8 --vae_n_latent 512 --n_samples 4 --steps 3000
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import Ds_point_sampled_already
from src.Vae import Vae, normalize_pc
from src.vae_up import VaeUp
from src.Diffusion import CosineSchedule, LatentPointDenoiser


@torch.no_grad()
def extract_latents(vae, points):
    """Média do posterior (latente limpo), igual ao train_diffusion.py."""
    x_norm  = normalize_pc(points)
    mu_g, _ = vae.global_encoder(x_norm)
    mu_l, _ = vae.local_encoder(x_norm, mu_g)
    return mu_g, mu_l


def main():
    p = argparse.ArgumentParser()
    # VAE
    p.add_argument("--vae_ckpt",        default="checkpoints/best.pt")
    p.add_argument("--data_root",       default="point_clouds")
    p.add_argument("--vae_type",        default="up", choices=["up", "base"])
    p.add_argument("--vae_latent_dim",  type=int, default=8)
    p.add_argument("--vae_n_latent",    type=int, default=512)
    p.add_argument("--vae_style_dim",   type=int, default=256)
    p.add_argument("--vae_in_channels", type=int, default=6)
    # denoiser (DiT) — mesmos defaults do DiffusionConfig
    p.add_argument("--point_hidden",    type=int, default=256)
    p.add_argument("--point_layers",    type=int, default=8)
    p.add_argument("--point_heads",     type=int, default=8)
    p.add_argument("--point_mlp_ratio", type=float, default=4.0)
    p.add_argument("--T",               type=int, default=1000)
    # overfit
    p.add_argument("--n_samples", type=int, default=4,    help="quantas amostras do batch fixar")
    p.add_argument("--steps",     type=int, default=3000)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ---- VAE congelado ----
    if args.vae_type == "up":
        vae = VaeUp(latent_dim=args.vae_latent_dim, style_dim=args.vae_style_dim,
                    in_channels=args.vae_in_channels, n_latent=args.vae_n_latent).to(device)
    else:
        vae = Vae(latent_dim=args.vae_latent_dim, style_dim=args.vae_style_dim,
                  in_channels=args.vae_in_channels).to(device)
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint nao encontrado: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])
    vae.eval()
    for pm in vae.parameters():
        pm.requires_grad_(False)
    print(f"VAE carregado de {args.vae_ckpt} (congelado)")

    # ---- UM batch fixo ----
    ds = Ds_point_sampled_already(root=args.data_root, augment=False)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True)
    points, _ = next(iter(loader))
    points = points.to(device)
    print(f"Batch fixo: {points.shape[0]} amostras")

    z_g, z_l = extract_latents(vae, points)   # (B,S), (B,N,L)

    # ---- Normalizacao por-dim (mesma semantica do LatentNormalizer),
    #      usando as stats do proprio batch, pra loss ficar comparavel ao 0.36 ----
    zl_flat = z_l.reshape(-1, z_l.shape[-1])
    zl_mean, zl_std = zl_flat.mean(0), zl_flat.std(0).clamp(min=1e-6)
    zg_mean, zg_std = z_g.mean(0),     z_g.std(0).clamp(min=1e-6)
    z_l = (z_l - zl_mean) / zl_std
    z_g = (z_g - zg_mean) / zg_std
    print(f"z_l normalizado: mean={z_l.mean():.3f} std={z_l.std():.3f}  shape={tuple(z_l.shape)}")

    # ---- Denoiser DiT + schedule ----
    sch = CosineSchedule(T=args.T).to(device)
    dn  = LatentPointDenoiser(
        point_dim=args.vae_latent_dim, style_dim=args.vae_style_dim,
        hidden=args.point_hidden, n_layers=args.point_layers,
        n_heads=args.point_heads, mlp_ratio=args.point_mlp_ratio, T=args.T,
    ).to(device)
    n_params = sum(pm.numel() for pm in dn.parameters())
    print(f"LatentPointDenoiser (DiT): {n_params:,} params\n")

    opt = torch.optim.AdamW(dn.parameters(), lr=args.lr)
    print(f"Baseline trivial N(0,I) = mean(acp) = {sch.acp.mean().item():.4f}")
    print(f"Treinando overfit por {args.steps} steps...\n")

    dn.train()
    best = float("inf")
    for step in range(args.steps):
        B = z_l.shape[0]
        t = torch.randint(0, sch.T, (B,), device=device)
        xt, noise = sch.q_sample(z_l, t)
        pred = dn(xt, t, z_g)
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(dn.parameters(), 1.0)
        opt.step()

        best = min(best, loss.item())
        if step % 100 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss_point={loss.item():.4f}  (melhor={best:.4f})")

    print("\n=== Veredicto ===")
    if best < 0.10:
        print(f"  loss chegou a {best:.4f} (<0.10) -> DENOISER OK.")
        print("  => 0.36 no dataset cheio e PISO DE DADOS. Ataque o VAE (KL / ancoras xyz).")
    elif best < 0.25:
        print(f"  loss chegou a {best:.4f} -> denoiser aprende, mas com dificuldade.")
        print("  => Provavel mistura: latente pobre + tuning. Cheque recon do VAE.")
    else:
        print(f"  loss travou em {best:.4f} mesmo overfitando -> GARGALO/BUG no denoiser.")
        print("  => O problema esta no modelo/otimizacao, nao nos dados.")


if __name__ == "__main__":
    main()
