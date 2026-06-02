import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model
from tqdm import tqdm

# ── Helpers ───────────────────────────────────────────────────────────────────


def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    """
    Salva os point clouds original e reconstruído como .ply.
    Espera tensores de shape (N, 6) — XYZ (3) + normais/features (3).
    """
    os.makedirs("reconstructions", exist_ok=True)

    orig = original_tensor.detach().cpu().numpy()
    recon = reconstructed_tensor.detach().cpu().numpy()

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])
    normais_orig = orig[:, 3:]
    mags = np.linalg.norm(normais_orig, axis=-1, keepdims=True) + 1e-8
    pcd_orig.normals = o3d.utility.Vector3dVector(normais_orig / mags)

    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])

    path_orig = f"reconstructions/epoch_{epoch + 1}_{class_name}_ORIGINAL.ply"
    path_recon = f"reconstructions/epoch_{epoch + 1}_{class_name}_RECONSTRUCTED.ply"
    o3d.io.write_point_cloud(path_orig, pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)

    print(f"\n[INFO] Amostra da Época {epoch + 1} salva em /reconstructions")


def log_codebook_usage(model: DualBranchPointVQVAE, xyz: torch.Tensor, device):
    """
    Calcula quantos vetores do codebook estão sendo usados (coverage).
    Útil para detectar 'codebook collapse' — sintoma comum em VQ-VAEs.
    """
    with torch.no_grad():
        out = model(xyz.to(device))
        indices = out["indices"]  # (B,)
        unique = indices.unique().numel()
        total = model.n_embeddings
    return unique, total


# ── Training loop ─────────────────────────────────────────────────────────────


def train_vae_gan_single_batch(
    model: DualBranchPointVAE,
    dataset,
    num_epochs: int = 1000,
    batch_size: int = 16,
    lr_g: float = 1e-3,
    lr_d: float = 1e-4,  # D aprende mais devagar que G
    d_steps: int = 1,  # passos do D por passo do G
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fixed_class_names, fixed_pcd = next(iter(dataloader))
    fixed_xyz = fixed_pcd.to(device)
    fixed_class_name = fixed_class_names[0]

    # Parâmetros do gerador (tudo exceto discriminador e embedding EMA)
    gen_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("discriminator")
        and not n.startswith("bottleneck.embedding")
    ]
    # Parâmetros do discriminador
    disc_params = list(model.discriminator.parameters())

    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc_params, lr=lr_d, betas=(0.5, 0.9))

    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, mode="min", factor=0.5, patience=50
    )

    loop = tqdm(range(num_epochs), desc="VQ-VAE + GAN")

    for epoch in loop:
        print(f"epoch {epoch}")
        model.train()

        # ── Passo D (discriminador) ───────────────────────────────────────
        for _ in range(d_steps):
            with torch.no_grad():
                out_d = model(fixed_xyz)  # forward sem gradiente pro G
            d_loss = model.discriminator_loss(out_d["recon"], fixed_xyz)

            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc_params, 1.0)
            opt_d.step()

        # ── Passo G (gerador = encoder + VQ + decoder) ────────────────────
        out = model(fixed_xyz)
        g_loss_dict = model.loss(out, fixed_xyz)

        opt_g.zero_grad()
        g_loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()
        sched_g.step(g_loss_dict["total"])

        loop.set_postfix(
            G=f"{g_loss_dict['total'].item():.4f}",
            cd=f"{g_loss_dict['cd'].item():.4f}",
            vq=f"{g_loss_dict['codebook_loss'].item():.4f}",
            adv_g=f"{g_loss_dict['adv_g'].item():.4f}",
            D=f"{d_loss.item():.4f}",
        )

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            save_before_after(fixed_xyz[0], out["recon"][0], epoch, fixed_class_name)

            used, total = log_codebook_usage(model, fixed_xyz, device)
            coverage = 100.0 * used / total
            tqdm.write(
                f"  [Codebook] {used}/{total} ({coverage:.1f}%) "
                + ("✓" if coverage > 50 else "⚠ collapse!")
            )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "epoch": num_epochs,
            "model_state": model.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
        },
        "checkpoints/dual_branch_vqvae_gan.pt",
    )
    print("\n[SUCESSO] Treino VQ-VAE + GAN concluído.")


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranchPointVAE(
        d_model=512,
        latent_dim=512,
        n_embeddings=512,
        n_out=2048,
        enc_depth=10,
        dec_depth=10,
        n_heads=8,
        commitment_cost=0.25,
        ema_update=True,
        adv_weight=0.1,  # começa conservador — aumente se o D convergir rápido
    ).to(device)

    dataset = Ds_point_sampled(Ds_point_model())
    
    train_vqvae_gan_single_batch(model, dataset, num_epochs=1000, batch_size=1)
