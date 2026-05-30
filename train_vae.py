import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Vae import VAE
from src.dataset import Ds_point_sampled, Ds_point_model
from tqdm import tqdm


def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    """
    Salva a nuvem de pontos original e a reconstruída no formato .ply do Open3D.
    Espera tensores no formato de canais [6, 2048]
    """
    os.makedirs("reconstructions", exist_ok=True)

    # 1. Converte de volta para NumPy e remove do gradiente/GPU -> formato (6, 2048)
    orig = original_tensor.detach().cpu().numpy()
    recon = reconstructed_tensor.detach().cpu().numpy()

    # 2. Rotaciona de volta para o formato padrão do Open3D -> formato (2048, 6)
    orig = orig.T
    recon = recon.T

    # 3. Cria o objeto Open3D para a nuvem Original
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])
    pcd_orig.normals = o3d.utility.Vector3dVector(orig[:, 3:])

    # 4. Cria o objeto Open3D para a nuvem Reconstruída
    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])

    # 5. Salva os arquivos .ply
    path_orig = f"reconstructions/epoch_{epoch+1}_{class_name}_ORIGINAL.ply"
    path_recon = f"reconstructions/epoch_{epoch+1}_{class_name}_RECONSTRUCTED.ply"

    o3d.io.write_point_cloud(path_orig, pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)
    print(f"\n[INFO] Amostra salva em {path_orig} e {path_recon}")


def train_vae(model, dataset, num_epochs=10, batch_size=32, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        # Variáveis auxiliares para salvar o último item processado da época
        last_pcd_input = None
        last_recon_pcd = None
        last_class_name = "unknown"

        for class_name, pcd in loop:
            pcd = pcd.to(device)

            # Passa de [16, 2048, 6] para -> [16, 6, 2048]
            pcd_input = pcd.permute(0, 2, 1)

            optimizer.zero_grad()

            recon_pcd, mu, logvar = model(pcd_input)
            recon_pcd = recon_pcd.squeeze(-1)

            recon_loss = reconstruction_loss_fn(recon_pcd, pcd_input)
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Guarda a referência do batch atual para salvar no fim da época
            last_pcd_input = pcd_input
            last_recon_pcd = recon_pcd
            last_class_name = class_name[
                0
            ]  # Pega o nome da classe do primeiro item do batch

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Finalizadas, Loss Média: {avg_loss:.4f}")

        # Salva o "Antes e Depois" da primeira amostra (índice 0) do último batch da época
        if last_pcd_input is not None:
            save_before_after(
                last_pcd_input[0], last_recon_pcd[0], epoch, last_class_name
            )


if __name__ == "__main__":
    # Silencia os avisos repetitivos do Open3D na leitura interna do dataset
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    vae = VAE(input_dim=6, latent_dim=128).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = Ds_point_sampled(Ds_point_model())

    train_vae(vae, dataset, num_epochs=10, batch_size=16, learning_rate=1e-3)
