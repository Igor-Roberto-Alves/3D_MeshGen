import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.VQ_VAE import DualBranchPointVQVAE
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


def train_vqvae_single_batch(
    model: DualBranchPointVQVAE,
    dataset,
    num_epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("[INFO] Capturando um único batch fixo para teste de overfitting...")
    fixed_batch = next(iter(dataloader))
    fixed_class_names, fixed_pcd = fixed_batch
    fixed_xyz = fixed_pcd.to(device)
    fixed_class_name = fixed_class_names[0]

    # Com EMA ativado, os parâmetros do codebook não recebem gradiente —
    # por isso filtramos antes de passar pro optimizer para evitar warnings.
    trainable = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("bottleneck.embedding")
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    # Scheduler: reduz LR quando a loss estabiliza (padrão para VQ-VAEs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    loop = tqdm(range(num_epochs), desc="Treinando VQ-VAE (batch fixo)")

    for epoch in loop:
        model.train()
        optimizer.zero_grad()

        out = model(fixed_xyz)
        loss_dict = model.loss(out, fixed_xyz)

        loss_dict["total"].backward()

        # Gradient clipping — evita explosão em transformers
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        optimizer.step()
        scheduler.step(loss_dict["total"])

        loop.set_postfix(
            loss=f"{loss_dict['total'].item():.4f}",
            cd=f"{loss_dict['cd'].item():.4f}",
            vq=f"{loss_dict['codebook_loss'].item():.4f}",
        )

        # A cada 20 épocas: salva reconstrução + monitora uso do codebook
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            save_before_after(
                fixed_xyz[0],
                out["recon"][0],
                epoch,
                fixed_class_name,
            )

            used, total = log_codebook_usage(model, fixed_xyz, device)
            coverage = 100.0 * used / total
            tqdm.write(
                f"  [Codebook] {used}/{total} vetores ativos ({coverage:.1f}%) "
                + ("✓" if coverage > 50 else "⚠ possível collapse!")
            )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "epoch": num_epochs,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "avg_loss": loss_dict["total"].item(),
        },
        "checkpoints/dual_branch_vqvae_OVERFIT_BATCH.pt",
    )
    print("\n[SUCESSO] Treino de overfitting VQ-VAE concluído.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranchPointVQVAE(
        d_model=512,
        latent_dim=512,
        n_embeddings=512,  # tamanho do codebook
        n_out=2048,
        enc_depth=4,
        dec_depth=4,
        n_heads=8,
        commitment_cost=0.25,  # β da commitment loss
        ema_update=True,  # atualização estável via EMA
    ).to(device)

    print("[INFO] Inicializando o Dataset...")
    dataset = Ds_point_sampled(Ds_point_model())

    train_vqvae_single_batch(
        model,
        dataset,
        num_epochs=1000,
        batch_size=16,
        learning_rate=1e-3,
    )
