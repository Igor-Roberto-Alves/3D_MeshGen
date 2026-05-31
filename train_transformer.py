import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model
from tqdm import tqdm


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_before_after(original_tensor, reconstructed_tensor, epoch, class_name):
    """
    Saves the original and reconstructed point clouds as .ply files.
    Expects tensors of shape (N, 6) — spatial (3) + features/normals (3).
    """
    os.makedirs("reconstructions", exist_ok=True)

    orig  = original_tensor.detach().cpu().numpy()    # (N, 6)
    recon = reconstructed_tensor.detach().cpu().numpy()  # (N, 6)

    # Process original point cloud
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])   # Slice out X, Y, Z
    
    # Normalização visual para garantir que o Open3D renderize o sombreamento idêntico ao recon
    normais_orig = orig[:, 3:]
    mags = np.linalg.norm(normais_orig, axis=-1, keepdims=True) + 1e-8
    pcd_orig.normals = o3d.utility.Vector3dVector(normais_orig / mags)

    # Process reconstructed point cloud
    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3]) # Slice out X, Y, Z
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:]) # Normais do decoder já vêm unitárias

    path_orig  = f"reconstructions/epoch_{epoch + 1}_{class_name}_ORIGINAL.ply"
    path_recon = f"reconstructions/epoch_{epoch + 1}_{class_name}_RECONSTRUCTED.ply"
    o3d.io.write_point_cloud(path_orig,  pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)

    print(f"\n[INFO] Amostra da Época {epoch + 1} salva com sucesso em /reconstructions")

# ── Training loop ─────────────────────────────────────────────────────────────

def train_vae_single_batch(
    model: DualBranchPointVAE,
    dataset,
    num_epochs: int  = 200,
    batch_size: int  = 16,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # DataLoader configurado de forma limpa
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("[INFO] Capturando um único batch fixo para teste de Overfitting...")
    fixed_batch = next(iter(dataloader))
    fixed_class_names, fixed_pcd = fixed_batch
    
    fixed_xyz = fixed_pcd.to(device)
    fixed_class_name = fixed_class_names[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loop = tqdm(range(num_epochs), desc="Treinando Lote Único")

    for epoch in loop:
        model.train()
        optimizer.zero_grad()
        
        # O modelo processa o lote e devolve o dicionário completo estruturado pelo novo Vae.py
        out = model(fixed_xyz)              
        loss_dict = model.loss(out, fixed_xyz)

        loss_dict["total"].backward() 
        optimizer.step()

        # Atualiza o postfix com os valores escalares limpos
        loop.set_postfix(
            loss=f"{loss_dict['total'].item():.4f}",
            cd=f"{loss_dict['cd'].item():.4f}",
            kl=f"{loss_dict['kl'].item():.4f}",
        )

        # Salva o progresso a cada 20 épocas
        if (epoch + 1) % 20 == 0 or epoch == 0:
            save_before_after(
                fixed_xyz[0],   
                out["recon"][0],       
                epoch,
                fixed_class_name,
            )

    # Persistência final estável do teste de sanidade geométrico
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "epoch":       num_epochs,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "avg_loss":    loss_dict["total"].item(),
        },
        "checkpoints/dual_branch_vae_OVERFIT_BATCH.pt",
    )
    print("\n[SUCESSO] Treino de Overfitting concluído sem erros.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Suprime mensagens excessivas do Open3D para não quebrar a UI do tqdm
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualBranchPointVAE(
        d_model   = 512,
        latent_dim= 512,
        n_out     = 2048,
        enc_depth = 4,
        dec_depth = 4,
        n_heads   = 8,
        beta      = 0.0, # Beta estrito em zero força o modelo a ignorar o gargalo do VAE
    ).to(device)

    print("[INFO] Inicializando o Dataset...")
    dataset = Ds_point_sampled(Ds_point_model())
    
    # Executa o treino curto de 10 épocas para validação de gradiente
    train_vae_single_batch(
        model,
        dataset,
        num_epochs    = 1000, 
        batch_size    = 16,
        learning_rate = 1e-3,
    )
