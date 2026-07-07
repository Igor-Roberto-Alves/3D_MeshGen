"""
recon_vae.py
------------
Reconstrução do VAE congelado + estrutura do latente.
Agora suporta VaeFlat e VaePointnet.
Gera gráficos visuais das reconstruções e da distribuição do espaço latente 
para relatórios finais.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
from torch.utils.data import DataLoader

from src.dataset import Ds_point_sampled_already
from src.Vae import normalize_pc
from src.metric import chamfer_distance_knn, f_score

from train_diffusion_flat_noclass import load_vae, list_files_flat, list_files_with_labels, make_split_flat, make_split, load_clouds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt",        default="ckpt_pointvae/exp0_baseline/best.pt")
    p.add_argument("--data_root",       default=None)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--n_batches",       type=int, default=8)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--plot_samples",    type=int, default=4, help="quantas amostras plotar")
    p.add_argument("--plot_out",        default="recon_vae.png")
    p.add_argument("--plot_latent",     default="latent_stats.png", help="Gráfico das dimensões ativas")
    args = p.parse_args()

    FSCORE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1]

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ---- VAE congelado ----
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint não encontrado: {args.vae_ckpt}")
        
    vae, vae_args = load_vae(args.vae_ckpt, device)
    flat = vae_args.get("arch", "") == "flat"
    
    print(f"VAE carregado de {args.vae_ckpt} (congelado) | Arch: {vae_args.get('arch', 'unknown')}\n")

    data_root = args.data_root or vae_args["data_root"]
    
    if flat:
        files, _ = list_files_flat(data_root)
    else:
        files, _ = list_files_with_labels(data_root, vae_args.get("category", "all"))
        
    # Vamos pegar apenas as primeiras N nuvens para teste rápido
    files = files[:args.batch_size * args.n_batches]
    print("Carregando nuvens...")
    clouds = load_clouds(files, vae_args.get("in_dim", 3))
    
    loader = DataLoader(clouds, batch_size=args.batch_size, shuffle=False)

    cds, zl_all = [], []
    fss = {th: [] for th in FSCORE_THRESHOLDS}
    plot_gt = None
    plot_recon = None
    
    for bi, points in enumerate(loader):
        points = points.to(device)
        if flat:
            points = normalize_pc(points)
            
        target_xyz = points[..., :3]

        # Reconstrução DETERMINÍSTICA: média do posterior
        if flat:
            mu, _ = vae.encoder(points)
            xyz_out = vae.decoder(mu)
        else:
            mu, _, _ = vae.encoder(points.permute(0, 2, 1))
            xyz_out = vae.decoder(mu)

        n = min(xyz_out.shape[1], target_xyz.shape[1])
        cd, _, _ = chamfer_distance_knn(xyz_out[:, :n], target_xyz[:, :n], reduce="none")
        cds.append(cd.cpu())
        for th in FSCORE_THRESHOLDS:
            fs = f_score(xyz_out[:, :n], target_xyz[:, :n], threshold=th, reduce="none")
            fss[th].append(fs["f_score"].cpu())
            
        zl_all.append(mu.reshape(-1, mu.shape[-1]).cpu())

        if bi == 0 and args.plot_samples > 0:
            k = min(args.plot_samples, points.shape[0])
            plot_gt    = target_xyz[:k].cpu()
            plot_recon = xyz_out[:k].cpu()

    cds = torch.cat(cds)
    zl  = torch.cat(zl_all)   # (n_pontos_total, latent_dim)

    # Espaçamento típico
    ref = target_xyz[0, :n]
    d_intra = torch.cdist(ref, ref)
    d_intra.fill_diagonal_(float("inf"))
    nn_intra = d_intra.min(dim=1).values.mean().item()

    print("=== Reconstrução do VAE (determinística) ===")
    print(f"  Chamfer:  mean={cds.mean():.5f}  median={cds.median():.5f}  max={cds.max():.5f}")
    for th in FSCORE_THRESHOLDS:
        print(f"  F-Score@{th:<4}: mean={torch.cat(fss[th]).mean():.4f}")
    print(f"  (espaçamento médio entre vizinhos na nuvem real ~ {nn_intra:.4f})")
    print(f"  (amostras avaliadas: {cds.shape[0]})\n")

    # ---- Estrutura do latente ----
    std_per_dim = zl.std(dim=0)
    std_sorted, _ = torch.sort(std_per_dim, descending=True)
    
    print("=== Estrutura do latente ===")
    print(f"  latent_dim = {zl.shape[1]}")
    active_dims = (std_per_dim > 0.05).sum().item()
    print(f"  Dimensões ativas (std > 0.05): {active_dims} / {zl.shape[1]}")
    print(f"  std por-dim (Top 10): {[round(s.item(),3) for s in std_sorted[:10]]}")

    # ---- Plot Input vs Recon ----
    if plot_gt is not None:
        _plot_recon(plot_gt, plot_recon, args.plot_out)
        print(f"Plot de reconstrução salvo em: {args.plot_out}")
        
    # ---- Plot Latent Distribution (Para o Relatório) ----
    _plot_latent_stats(std_sorted, active_dims, args.plot_latent)
    print(f"Gráfico de dimensões latentes salvo em: {args.plot_latent}\n")

def _plot_recon(gt, recon, out_path):
    k = gt.shape[0]
    fig = plt.figure(figsize=(10, 4 * k))
    fig.suptitle('Reconstrução VAE (Esquerda: Original, Direita: Gerado)', fontsize=16)
    for i in range(k):
        for col, (pts, title) in enumerate([(gt[i], "Ground Truth"), (recon[i], "Reconstrução")]):
            ax = fig.add_subplot(k, 2, i * 2 + col + 1, projection="3d")
            p = pts.numpy()
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2,
                       c=("green" if col == 0 else "tab:blue"), alpha=0.6)
            ax.set_title(f"Amostra #{i} - {title}", fontsize=12)
            ax.set_box_aspect([1, 1, 1])
            ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def _plot_latent_stats(std_sorted, active_dims, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(std_sorted))
    
    # Destaca as dimensões ativas
    colors = ['tab:blue' if s > 0.05 else 'tab:gray' for s in std_sorted]
    ax.bar(x, std_sorted.numpy(), color=colors, width=1.0)
    
    ax.set_title("Variância do Espaço Latente por Dimensão (Posterior Collapse)", fontsize=14)
    ax.set_xlabel("Índice da Dimensão Latente (Ordenado por Variância)", fontsize=12)
    ax.set_ylabel("Desvio Padrão ($\sigma$ das médias $\mu$)", fontsize=12)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Limiar de Atividade (0.05)')
    
    ax.text(0.95, 0.95, f'Dimensões Ativas: {active_dims} / {len(std_sorted)}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
