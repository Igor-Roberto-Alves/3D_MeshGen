"""
generate_gaussian.py
---------------------
Modelo generativo alternativo, SEM difusão, usando só o VAE congelado.

Ideia: em vez de aprender p(z_g|classe) e p(z_l|z_g) com um denoiser (que ainda
não convergiu), estimamos essas distribuições diretamente dos latentes reais:

  1. z_g ~ N(mean_cls, diag(std_cls)^2)   -- Gaussiana por classe (estilo global)
  2. z_l -- k-NN em z_g real da mesma classe: pega o z_l de exemplos reais cujo
     z_g está perto do z_g amostrado, faz uma média ponderada por distância
     entre os k vizinhos + ruído leve, preservando estrutura local real em vez
     de tentar modelar 512*3 dims direto (inviável com poucos dados).
  3. Decodifica com o decoder do VAE (congelado).

Isso não tem loss pra travar e não depende de sampling DDPM de 1000 passos --
é uma estatística direta sobre o cache de latentes. Serve de Plano B rápido
enquanto a difusão treina, ou de fallback caso ela não convirja a tempo.

Uso:
  python generate_gaussian.py --vae_ckpt checkpoints/best.pt --data_root point_clouds \
         --n_gen 16 --k 5 --noise_scale 0.15
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.dataset import Ds_point_sampled_already, Ds_point_model
from src.Vae import Vae, normalize_pc
from src.vae_up import VaeUp
from src.metric import generative_metrics


@torch.no_grad()
def extract_latents(vae, points):
    x_norm  = normalize_pc(points)
    mu_g, _ = vae.global_encoder(x_norm)
    mu_l, _ = vae.local_encoder(x_norm, mu_g)
    return mu_g, mu_l


@torch.no_grad()
def sample_gaussian_knn(z_g_real, z_l_real, n_gen, k, noise_scale, device):
    """
    z_g_real: (M, style_dim)   z_l_real: (M, N, latent_dim)  -- reais, 1 classe

    Retorna (z_g_gen, z_l_gen) amostrados: z_g de uma Gaussiana diagonal ajustada
    aos dados reais; z_l por interpolação ponderada dos k vizinhos reais mais
    próximos (em z_g) do z_g amostrado, mais um ruído leve pra não colar 100%
    em pontos de treino.
    """
    mean_g = z_g_real.mean(0)
    std_g  = z_g_real.std(0).clamp(min=1e-6)

    z_g_gen = mean_g + std_g * torch.randn(n_gen, mean_g.shape[0], device=device)

    # k-NN em z_g real para cada z_g amostrado
    d = torch.cdist(z_g_gen, z_g_real)                      # (n_gen, M)
    knn_dist, knn_idx = d.topk(min(k, z_g_real.shape[0]), dim=1, largest=False)
    w = (-knn_dist).softmax(dim=1)                           # pesos: mais perto = mais peso

    # média ponderada dos z_l dos vizinhos: (n_gen, k, N, L) * (n_gen, k, 1, 1)
    z_l_neighbors = z_l_real[knn_idx]                        # (n_gen, k, N, L)
    z_l_gen = (z_l_neighbors * w.view(*w.shape, 1, 1)).sum(dim=1)  # (n_gen, N, L)

    # ruído leve proporcional ao std local do latente, pra gerar variação
    # em vez de reproduzir literalmente os vizinhos
    local_std = z_l_real.std(dim=0, keepdim=True)            # (1, N, L)
    z_l_gen = z_l_gen + noise_scale * local_std * torch.randn_like(z_l_gen)

    return z_g_gen, z_l_gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt",        default="checkpoints/best.pt")
    p.add_argument("--data_root",       default="point_clouds")
    p.add_argument("--vae_type",        default="up", choices=["up", "base"])
    p.add_argument("--vae_latent_dim",  type=int, default=8)
    p.add_argument("--vae_n_latent",    type=int, default=512)
    p.add_argument("--vae_style_dim",   type=int, default=256)
    p.add_argument("--vae_in_channels", type=int, default=6)
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--val_split",       type=float, default=0.2, help="fracao real reservada como referencia p/ metricas")
    p.add_argument("--n_gen",           type=int, default=16, help="quantas amostras gerar por classe")
    p.add_argument("--k",               type=int, default=5,  help="vizinhos usados na interpolacao de z_l")
    p.add_argument("--noise_scale",     type=float, default=0.15, help="ruido extra em z_l, fracao do std local")
    p.add_argument("--eval_points",     type=int, default=1024)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--plot_out",        default="generate_gaussian.png")
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
    print(f"VAE carregado de {args.vae_ckpt} (congelado)\n")

    # ---- Dataset completo, extrai latentes de tudo ----
    ds     = Ds_point_sampled_already(root=args.data_root, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    name_map    = Ds_point_model.map()
    idx_to_name = {idx: name_map.get(cid, cid) for cid, idx in ds.class_to_idx.items()}

    ZG, ZL, CLS, XYZ = [], [], [], []
    for points, cls in loader:
        points = points.to(device)
        z_g, z_l = extract_latents(vae, points)
        ZG.append(z_g.cpu()); ZL.append(z_l.cpu()); CLS.append(cls)
        XYZ.append(normalize_pc(points)[..., :3].cpu())
    ZG, ZL, CLS, XYZ = torch.cat(ZG), torch.cat(ZL), torch.cat(CLS), torch.cat(XYZ)
    print(f"Latentes extraidos: z_g {tuple(ZG.shape)}  z_l {tuple(ZL.shape)}  ({CLS.numel()} amostras)\n")

    # ---- split por classe: parte "fit" (estatisticas) / parte "ref" (metricas) ----
    plot_gt = plot_gen = None
    results = {}
    for cls_idx in sorted(idx_to_name.keys()):
        mask = (CLS == cls_idx)
        idx  = torch.where(mask)[0]
        idx  = idx[torch.randperm(idx.numel())]
        n_val = max(1, int(idx.numel() * args.val_split))
        ref_idx, fit_idx = idx[:n_val], idx[n_val:]
        if fit_idx.numel() < args.k + 1:
            print(f"[{idx_to_name[cls_idx]}] poucas amostras ({fit_idx.numel()}), pulando.")
            continue

        z_g_fit = ZG[fit_idx].to(device)
        z_l_fit = ZL[fit_idx].to(device)
        ref_xyz = XYZ[ref_idx].to(device)

        z_g_gen, z_l_gen = sample_gaussian_knn(
            z_g_fit, z_l_fit, args.n_gen, args.k, args.noise_scale, device
        )
        gen_xyz = vae.decoder(z_l_gen, z_g_gen).float()

        n_ref = min(ref_xyz.shape[0], gen_xyz.shape[0])
        g, r = gen_xyz, ref_xyz
        if args.eval_points < g.shape[1]:
            gi = torch.randperm(g.shape[1], device=device)[:args.eval_points]
            ri = torch.randperm(r.shape[1], device=device)[:args.eval_points]
            g, r = g[:, gi], r[:, ri]

        m = generative_metrics(g, r, chunk=4)
        results[idx_to_name[cls_idx]] = {k: v.item() for k, v in m.items()}
        print(f"[{idx_to_name[cls_idx]}]  MMD-CD={m['mmd']:.5f}  COV-CD={m['cov']:.3f}  "
              f"1-NNA-CD={m['nna_1nn']:.3f}  (n_fit={fit_idx.numel()}  n_ref={n_ref}  n_gen={args.n_gen})")

        if plot_gt is None:
            k_show = min(4, ref_xyz.shape[0], gen_xyz.shape[0])
            plot_gt  = ref_xyz[:k_show].cpu()
            plot_gen = gen_xyz[:k_show].cpu()

    if results:
        mmd_mean  = sum(v["mmd"] for v in results.values()) / len(results)
        cov_mean  = sum(v["cov"] for v in results.values()) / len(results)
        nna_mean  = sum(v["nna_1nn"] for v in results.values()) / len(results)
        print(f"\n=== Media geral ===  MMD-CD={mmd_mean:.5f}  COV-CD={cov_mean:.3f}  1-NNA-CD={nna_mean:.3f}")
        print("(0.5 = ideal p/ 1-NNA; quanto mais perto de 0.5, mais indistinguivel do real)")

    if plot_gt is not None:
        _plot(plot_gt, plot_gen, args.plot_out)
        print(f"\nPlot salvo em: {args.plot_out}")


def _plot(gt, gen, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    k = gt.shape[0]
    fig = plt.figure(figsize=(6, 3 * k))
    for i in range(k):
        for col, (pts, title, color) in enumerate([
            (gt[i],  "real (ref)", "green"),
            (gen[i], "gerado",     "tab:orange"),
        ]):
            ax = fig.add_subplot(k, 2, i * 2 + col + 1, projection="3d")
            pv = pts.numpy()
            ax.scatter(pv[:, 0], pv[:, 1], pv[:, 2], s=2, c=color)
            ax.set_title(f"#{i} {title}", fontsize=9)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
