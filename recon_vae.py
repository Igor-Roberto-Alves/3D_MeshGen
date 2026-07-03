"""
recon_vae.py
------------
Reconstrucao do VAE congelado + estrutura do latente z_l.

Equivale a Secao 4 do analyze_vae_diffusion.ipynb, mas standalone (so precisa do
VAE, nao de um checkpoint de difusao — que nao carrega mais no DiT novo).

Responde a pergunta que disambigua o plato da loss_point em 0.36:
  - VAE reconstroi BEM  -> o latente carrega a estrutura da forma.
        => loss 0.36 travada = denoiser nao captura o que existe -> tunavel.
  - VAE reconstroi MAL  -> o latente e pobre.
        => 0.36 e teto do latente -> atacar o VAE (KL / ancoras xyz).

Tambem mede quanta ESTRUTURA ha em z_l (std por-dim + rank efetivo via PCA):
se z_l ~ ruido independente por dim, nao ha o que a difusao aprender.

Uso (mesmos args de VAE do treino):
  python recon_vae.py --vae_ckpt checkpoints/best.pt --data_root point_clouds \
         --vae_type up --vae_latent_dim 8 --vae_n_latent 512 --n_batches 8
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.dataset import Ds_point_sampled_already
from src.Vae import Vae, normalize_pc
from src.vae_up import VaeUp
from src.metric import chamfer_distance_knn, f_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt",        default="checkpoints/best.pt")
    p.add_argument("--data_root",       default="point_clouds")
    p.add_argument("--vae_type",        default="up", choices=["up", "base"])
    p.add_argument("--vae_latent_dim",  type=int, default=8)
    p.add_argument("--vae_n_latent",    type=int, default=512)
    p.add_argument("--vae_style_dim",   type=int, default=256)
    p.add_argument("--vae_in_channels", type=int, default=6)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--n_batches",       type=int, default=8)
    p.add_argument("--fscore_thresh",   type=float, default=0.01)
    p.add_argument("--device",          default="cuda")
    args = p.parse_args()

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
    print(f"VAE carregado de {args.vae_ckpt} (congelado)\n")

    ds = Ds_point_sampled_already(root=args.data_root, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    cds, fss, zl_all = [], [], []
    it = iter(loader)
    for _ in range(args.n_batches):
        try:
            points, _ = next(it)
        except StopIteration:
            break
        points = points.to(device)
        target_xyz = normalize_pc(points)[..., :3]

        # Reconstrucao DETERMINISTICA: media do posterior (sem amostrar ruido)
        x_norm  = normalize_pc(points)
        mu_g, _ = vae.global_encoder(x_norm)
        mu_l, _ = vae.local_encoder(x_norm, mu_g)
        xyz_out = vae.decoder(mu_l, mu_g)

        n = min(xyz_out.shape[1], target_xyz.shape[1])
        cd, _, _ = chamfer_distance_knn(xyz_out[:, :n], target_xyz[:, :n], reduce="none")
        fs = f_score(xyz_out[:, :n], target_xyz[:, :n], threshold=args.fscore_thresh, reduce="none")

        cds.append(cd.cpu())
        fss.append(fs["f_score"].cpu())
        zl_all.append(mu_l.reshape(-1, mu_l.shape[-1]).cpu())

    cds = torch.cat(cds)
    fss = torch.cat(fss)
    zl  = torch.cat(zl_all)   # (n_pontos_total, latent_dim)

    print("=== Reconstrucao do VAE (deterministica) ===")
    print(f"  Chamfer:  mean={cds.mean():.5f}  median={cds.median():.5f}  max={cds.max():.5f}")
    print(f"  F-Score@{args.fscore_thresh}: mean={fss.mean():.4f}  (1.0=perfeito)")
    print(f"  (amostras avaliadas: {cds.shape[0]})\n")

    # ---- Estrutura do latente z_l ----
    std_per_dim = zl.std(dim=0)
    zl_c = zl - zl.mean(dim=0, keepdim=True)
    # rank efetivo via entropia dos autovalores da covariancia (participation ratio)
    cov = (zl_c.T @ zl_c) / zl_c.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    part_ratio = (eig.sum() ** 2 / (eig ** 2).sum().clamp(min=1e-12)).item()

    print("=== Estrutura do latente z_l ===")
    print(f"  latent_dim = {zl.shape[1]}")
    print(f"  std por-dim: {[round(s.item(),3) for s in std_per_dim]}")
    print(f"  rank efetivo (participation ratio) = {part_ratio:.2f} de {zl.shape[1]}")
    print(f"  correlacao media |off-diag| = {cov.fill_diagonal_(0).abs().mean():.4f}\n")

    print("=== Como ler ===")
    print("  - F-Score alto (>~0.7) e Chamfer baixo => latente carrega estrutura")
    print("    => 0.36 travado e problema do DENOISER (tunavel).")
    print("  - F-Score baixo / Chamfer alto => latente pobre")
    print("    => 0.36 e teto do VAE (mexer em KL / ancoras xyz).")
    print("  - rank efetivo << latent_dim ou dims com std~0 => colapso posterior parcial.")


if __name__ == "__main__":
    main()
