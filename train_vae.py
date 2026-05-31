import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model
from tqdm import tqdm


def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    os.makedirs("reconstructions", exist_ok=True)

    # Coleta os dados direto no formato correto (N, 6)
    orig  = original_tensor.detach().cpu().numpy()    
    recon = reconstructed_tensor.detach().cpu().numpy()  

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])
    pcd_orig.normals = o3d.utility.Vector3dVector(orig[:, 3:])

    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])
    # 5. Salva os arquivos .ply
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_cd = total_kl = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        last_xyz_input = None
        last_recon     = None
        last_class     = "unknown"

        for class_name, pcd in loop:
            # Mantém o formato original (B, N, 6) exigido pelo DualBranchPointVAE
            xyz = pcd.to(device)          

            optimizer.zero_grad()
            
            # O modelo retorna um dicionário nativamente
            out = model(xyz)              
            
            # Usa o método de loss interno que calcula a Chamfer Distance corretamente!
            loss_dict = model.loss(out, xyz)
            
            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            total_cd   += loss_dict["cd"].item()
            total_kl   += loss_dict["kl"].item()

            loop.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                cd=f"{loss_dict['cd'].item():.4f}",
                kl=f"{loss_dict['kl'].item():.4f}",
            )

            last_xyz_input = xyz
            last_recon     = out["recon"]
            last_class     = class_name[0]

        n_batches  = len(dataloader)
        avg_loss   = total_loss / n_batches
        avg_cd     = total_cd   / n_batches
        avg_kl     = total_kl   / n_batches

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Done — "
            f"Loss Média: {avg_loss:.4f}  CD: {avg_cd:.4f}  KL: {avg_kl:.4f}"
        )

        # Salva amostras para checagem visual (com formato N, 6)
        if last_xyz_input is not None:
            save_before_after(
                last_xyz_input[0],   
                last_recon[0],       
                epoch,
                last_class,
            )

        # Salva o checkpoint do modelo atualizado
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "avg_loss":    avg_loss,
            },
            f"checkpoints/dual_branch_vae_epoch_{epoch + 1}.pt",
        )

# ── Entry point Corrigido ──────────────────────────────────────────────────

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instancia o modelo correto que você definiu no início do arquivo
    model = DualBranchPointVAE(
        d_model   = 384,
        latent_dim= 512,
        n_out     = 2048,
        enc_depth = 4,
        dec_depth = 4,
        n_heads   = 6,
        beta      = 1.0,
    ).to(device)

    dataset = Ds_point_sampled(Ds_point_model())

    # Roda o treino com a estrutura corrigida
    train_vae(
        model,
        dataset,
        num_epochs    = 200,
        batch_size    = 16,
        learning_rate = 1e-3,
    )