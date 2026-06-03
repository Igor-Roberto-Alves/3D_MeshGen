import os
import argparse
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model, Ds_point_sampled_already


def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    os.makedirs("reconstructions", exist_ok=True)

    orig = original_tensor.detach().cpu().numpy()
    recon = reconstructed_tensor.detach().cpu().numpy()

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])
    pcd_orig.normals = o3d.utility.Vector3dVector(orig[:, 3:])

    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])

    path_orig = f"reconstructions/epoch_{epoch+1}_{class_name}_ORIGINAL.ply"
    path_recon = f"reconstructions/epoch_{epoch+1}_{class_name}_RECONSTRUCTED.ply"

    o3d.io.write_point_cloud(path_orig, pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)
    print(f"\n[INFO] Amostra salva em {path_orig} e {path_recon}")


def train_vae(
    model: DualBranchPointVAE,
    dataset,
    num_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Recomendo AdamW para misturas de ViT + MLP
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_cd = total_kl = total_nl = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        last_xyz_input = None
        last_recon = None
        last_class = "unknown"

        for class_name, pcd in loop:
            xyz = pcd.to(device)
            optimizer.zero_grad()

            out = model(xyz)
            loss_dict = model.loss(out, xyz)

            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            total_cd += loss_dict["cd"].item()
            total_kl += loss_dict["kl"].item()
            total_nl += loss_dict["normal_loss"].item()

            loop.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                cd=f"{loss_dict['cd'].item():.4f}",
                kl=f"{loss_dict['kl'].item():.4f}",
                nl=f"{loss_dict['normal_loss'].item():.4f}",
            )

            last_xyz_input = xyz
            last_recon = out["recon"]
            last_class = class_name[0]

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_cd = total_cd / n_batches
        avg_kl = total_kl / n_batches
        avg_nl = total_nl / n_batches

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Done — "
            f"Loss Média: {avg_loss:.4f}  CD: {avg_cd:.4f}  KL: {avg_kl:.4f}  NL: {avg_nl:.4f}"
        )

        if last_xyz_input is not None:
            save_before_after(
                last_xyz_input[0],
                last_recon[0],
                epoch,
                last_class,
            )

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "avg_loss": avg_loss,
            },
            f"checkpoints/dual_branch_vae_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Dataset Training Loop.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=int,
        default=False,
        help="1 if dataset is charged, 0 if not",
    )

    # Instanciação Limpa combinando d_model=384 e o gargalo real de latent_dim=512
    model = DualBranchPointVAE(
        d_model=384,
        latent_dim=512,
        n_out=2048,
        enc_depth=8,
        dec_depth=4,
        n_heads=6,
        beta=0,  # Lembre de calibrar ou reduzir se o VAE tentar colapsar os pontos
    ).to(device)

    # Imprime a tabela de parâmetros do novo sistema híbrido para checagem estrutural
    model.report_parameters()

    if parser.parse_args().dataset:
        print("\n[INFO] Dataset already processed. Skipping point cloud sampling.")
        dataset = Ds_point_sampled_already()
    else:
        dataset = Ds_point_sampled(Ds_point_model())

    train_vae(
        model,
        dataset,
        num_epochs=200,
        batch_size=16,
        learning_rate=1e-3,
    )
