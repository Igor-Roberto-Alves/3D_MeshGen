"""
generate.py
-----------
Gera point clouds a partir do modelo treinado e calcula MMD / COV / 1-NNA.

Uso:
  python generate.py \
    --diff_ckpt  checkpoints_subset_diff/best.pt \
    --vae_ckpt   checkpoints_subset/best.pt \
    --data_root  point_clouds_subset \
    --n_gen      256 \
    --guidance   3.0 \
    --out_dir    generated_subset
"""

import argparse
import os
import torch
import numpy as np
import open3d as o3d

from src.Vae       import Vae, normalize_pc
from src.Diffusion import CosineSchedule, StyleDenoiser, LatentPointDenoiser
from src.metric    import generation_metrics, chamfer_distance_knn
from src.dataset   import Ds_point_sampled_already, Ds_point_model


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_ply(xyz: torch.Tensor, path: str):
    """xyz: (N, 3) CPU tensor → .ply file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.numpy().astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def denorm_zg(z_g, stats):
    return z_g if stats is None else z_g * stats["zg_std"] + stats["zg_mean"]


def denorm_zl(z_l, stats):
    return z_l if stats is None else z_l * stats["zl_std"] + stats["zl_mean"]


def load_models(args, device):
    # ── VAE ──────────────────────────────────────────────────
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_cfg  = vae_ckpt.get("config", {})
    vae = Vae(
        latent_dim  = vae_cfg.get("latent_dim",  3),
        style_dim   = vae_cfg.get("style_dim",   128),
        in_channels = vae_cfg.get("in_channels", 6),
    ).to(device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"VAE loaded from {args.vae_ckpt}")

    # ── Diffusion ─────────────────────────────────────────────
    diff_ckpt = torch.load(args.diff_ckpt, map_location=device, weights_only=False)
    diff_cfg  = diff_ckpt.get("config", {})

    schedule = CosineSchedule(T=diff_cfg.get("T", 1000)).to(device)

    style_dn = StyleDenoiser(
        style_dim   = vae.style_dim,
        num_classes = diff_cfg.get("num_classes",  55),
        hidden      = diff_cfg.get("style_hidden", 256),
        n_layers    = diff_cfg.get("style_layers", 4),
        T           = diff_cfg.get("T", 1000),
        cfg_dropout = 0.0,   # no dropout at inference
    ).to(device)

    point_dn = LatentPointDenoiser(
        point_dim  = vae.total_z_dim,
        style_dim  = vae.style_dim,
        hidden     = diff_cfg.get("point_hidden", 128),
        n_layers   = diff_cfg.get("point_layers", 6),
        T          = diff_cfg.get("T", 1000),
    ).to(device)

    # EMA weights are only trustworthy once the EMA has warmed up. On short
    # runs the EMA stays ~equal to the random init and produces cube-shaped
    # garbage, so --use_ema False falls back to the raw trained weights.
    if args.use_ema and "ema_style" in diff_ckpt:
        sd_style, sd_point = diff_ckpt["ema_style"], diff_ckpt["ema_point"]
        n_avg = int(diff_ckpt["ema_style"].get("n_averaged", torch.tensor(0)))
        print(f"Usando pesos EMA (n_averaged={n_avg})")
        if n_avg < 2000:
            print(f"  AVISO: EMA com poucos updates ({n_avg}) pode gerar lixo. "
                  f"Considere --use_ema False")
    else:
        sd_style, sd_point = diff_ckpt["style_dn"], diff_ckpt["point_dn"]
        print("Usando pesos RAW (treinados)")

    # AveragedModel wraps parameters under module.* and adds an n_averaged buffer
    def strip_module(sd):
        return {k.replace("module.", ""): v for k, v in sd.items() if k != "n_averaged"}

    style_dn.load_state_dict(strip_module(sd_style))
    point_dn.load_state_dict(strip_module(sd_point))
    style_dn.eval(); point_dn.eval()
    print(f"Diffusion loaded from {args.diff_ckpt}")

    # Latent standardisation stats (required to denormalise sampled latents).
    stats = diff_ckpt.get("latent_stats")
    if stats is not None:
        stats = {k: v.to(device) for k, v in stats.items()}
        print("Latent stats carregados (latentes serão desnormalizados)")
    else:
        print("AVISO: checkpoint sem latent_stats — geração pode ter escala errada "
              "(retreine com a versão nova do train_diffusion.py)")

    return vae, schedule, style_dn, point_dn, stats


# ─────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(vae, schedule, style_dn, point_dn, n_gen, guidance, device,
             num_points=2048, batch_size=16, stats=None):
    clouds = []
    total_zl = vae.total_z_dim
    num_classes = style_dn.uncond_idx  # = num_classes (before uncond token)

    for start in range(0, n_gen, batch_size):
        B   = min(batch_size, n_gen - start)
        cls = torch.randint(0, num_classes, (B,), device=device)

        uncond = style_dn.uncond(B, device)
        z_g    = schedule.sample(style_dn, (vae.style_dim,),
                                 condition=cls, uncond=uncond,
                                 guidance=guidance, device=device)
        z_l    = schedule.sample(point_dn, (num_points, total_zl),
                                 condition=z_g, device=device)
        z_g_dec = denorm_zg(z_g, stats)
        z_l_dec = denorm_zl(z_l, stats)
        xyz, _ = vae.decoder(z_l_dec, z_g_dec)
        clouds.append(xyz.float().cpu())
        print(f"  generated {start + B}/{n_gen}", end="\r")

    print()
    return torch.cat(clouds, dim=0)   # (n_gen, N, 3)


@torch.no_grad()
def generate_per_class(vae, schedule, style_dn, point_dn, class_ids,
                       n_per_class, guidance, device, num_points=2048, stats=None):
    """Generate n_per_class samples for each class in class_ids."""
    results = {}
    total_zl = vae.total_z_dim

    for cls_idx in class_ids:
        cls = torch.full((n_per_class,), cls_idx, device=device, dtype=torch.long)
        uncond = style_dn.uncond(n_per_class, device)
        z_g = schedule.sample(style_dn, (vae.style_dim,),
                              condition=cls, uncond=uncond,
                              guidance=guidance, device=device)
        z_l = schedule.sample(point_dn, (num_points, total_zl),
                              condition=z_g, device=device)
        z_g_dec = denorm_zg(z_g, stats)
        z_l_dec = denorm_zl(z_l, stats)
        xyz, _ = vae.decoder(z_l_dec, z_g_dec)
        results[cls_idx] = xyz.float().cpu()
    return results


# ─────────────────────────────────────────────────────────────
# Reference clouds from dataset
# ─────────────────────────────────────────────────────────────

def load_reference(data_root, n_ref, device):
    ds = Ds_point_sampled_already(root=data_root, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    refs = []
    for pts, _ in loader:
        refs.append(normalize_pc(pts)[..., :3].float())
        if sum(r.shape[0] for r in refs) >= n_ref:
            break
    ref = torch.cat(refs, dim=0)[:n_ref]
    return ref.to(device)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diff_ckpt",  default="checkpoints_subset_diff/best.pt")
    p.add_argument("--vae_ckpt",   default="checkpoints_subset/best.pt")
    p.add_argument("--data_root",  default="point_clouds_subset")
    p.add_argument("--n_gen",      type=int,   default=256)
    p.add_argument("--guidance",   type=float, default=3.0)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--out_dir",    default="generated_subset")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--num_points", type=int,   default=2048)
    p.add_argument("--save_ply",   action="store_true",
                   help="Save each generated cloud as a .ply file")
    p.add_argument("--use_ema",    type=lambda x: x.lower() != "false", default=True,
                   help="Use EMA weights (True) or raw trained weights (False). "
                        "Use False for short runs where the EMA hasn't warmed up.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────
    vae, schedule, style_dn, point_dn, stats = load_models(args, device)

    # ── Generate random samples ───────────────────────────────
    print(f"\nGerando {args.n_gen} amostras (guidance={args.guidance})...")
    gen = generate(vae, schedule, style_dn, point_dn,
                   n_gen=args.n_gen, guidance=args.guidance,
                   device=device, num_points=args.num_points,
                   batch_size=args.batch_size, stats=stats)
    print(f"  shape gerada: {gen.shape}")

    # ── Per-class generation ──────────────────────────────────
    idx_to_name = {idx: name for idx, (_, name) in
                   enumerate(Ds_point_model.map().items())}
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    # classes presentes no subconjunto
    present_names = ["airplane", "chair", "car"]
    present_ids   = [name_to_idx[n] for n in present_names if n in name_to_idx]

    print(f"\nGerando 8 amostras por classe: {present_names}")
    per_class = generate_per_class(vae, schedule, style_dn, point_dn,
                                   class_ids=present_ids, n_per_class=8,
                                   guidance=args.guidance, device=device,
                                   num_points=args.num_points, stats=stats)

    for cls_idx, xyz in per_class.items():
        name = idx_to_name.get(cls_idx, str(cls_idx))
        cls_dir = os.path.join(args.out_dir, name)
        os.makedirs(cls_dir, exist_ok=True)
        for i, cloud in enumerate(xyz):
            save_ply(cloud, os.path.join(cls_dir, f"sample_{i:02d}.ply"))
        print(f"  {name}: 8 .ply salvos em {cls_dir}/")

    # ── Save random samples ───────────────────────────────────
    if args.save_ply:
        rand_dir = os.path.join(args.out_dir, "random")
        os.makedirs(rand_dir, exist_ok=True)
        for i, cloud in enumerate(gen):
            save_ply(cloud, os.path.join(rand_dir, f"sample_{i:03d}.ply"))
        print(f"  {args.n_gen} amostras aleatórias salvas em {rand_dir}/")

    # ── Metrics ───────────────────────────────────────────────
    print(f"\nCarregando referências de {args.data_root}...")
    ref = load_reference(args.data_root, n_ref=args.n_gen, device=device)
    print(f"  ref shape: {ref.shape}")

    # resample to same N if needed
    N_gen = gen.shape[1]
    N_ref = ref.shape[1]
    if N_gen != N_ref:
        idx = torch.randperm(N_ref)[:N_gen]
        ref_eval = ref[:, idx, :].to(device)
    else:
        ref_eval = ref.to(device)

    gen_eval = gen.to(device)

    print("\nCalculando MMD / COV / 1-NNA  (pode demorar ~5 min)...")
    metrics = generation_metrics(gen_eval, ref_eval, batch_size=32)

    print("\n" + "=" * 45)
    print("  RESULTADOS DE GERAÇÃO")
    print("=" * 45)
    print(f"  MMD  (qualidade,  ↓ melhor): {metrics['mmd']:.6f}")
    print(f"  COV  (diversidade,↑ melhor): {metrics['cov']:.4f}  ({metrics['cov']*100:.1f}%)")
    print(f"  1-NNA(fidelidade, →0.5 bom): {metrics['nna']:.4f}")
    print("=" * 45)

    # Interpretação rápida
    nna = metrics["nna"]
    cov = metrics["cov"]
    if nna < 0.55 and cov > 0.40:
        status = "BOM — gerado e real são quase indistinguíveis"
    elif nna < 0.65 and cov > 0.25:
        status = "RAZOÁVEL — geração plausível com alguma diversidade"
    elif nna >= 0.85:
        status = "RUIM — geração muito diferente do real (mode collapse?)"
    else:
        status = "PARCIAL — cheque TensorBoard para diagnóstico"
    print(f"\n  Status: {status}")

    # Salvar métricas em txt
    result_path = os.path.join(args.out_dir, "metrics.txt")
    with open(result_path, "w") as f:
        f.write(f"MMD  = {metrics['mmd']:.6f}\n")
        f.write(f"COV  = {metrics['cov']:.4f}\n")
        f.write(f"NNA  = {metrics['nna']:.4f}\n")
        f.write(f"n_gen= {args.n_gen}\n")
        f.write(f"guidance= {args.guidance}\n")
        f.write(f"status= {status}\n")
    print(f"\n  Métricas salvas em: {result_path}")
    print(f"  .ply salvos em:    {args.out_dir}/")


if __name__ == "__main__":
    main()
