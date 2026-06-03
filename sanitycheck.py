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
    os.makedirs("reconstructions_sanity", exist_ok=True)

    # Coleta os dados direto no formato correto (N, 6)
    orig = original_tensor.detach().cpu().numpy()
    recon = reconstructed_tensor.detach().cpu().numpy()

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(orig[:, :3])
    pcd_orig.normals = o3d.utility.Vector3dVector(orig[:, 3:])

    pcd_recon = o3d.geometry.PointCloud()
    pcd_recon.points = o3d.utility.Vector3dVector(recon[:, :3])
    pcd_recon.normals = o3d.utility.Vector3dVector(recon[:, 3:])

    path_orig = f"reconstructions_sanity/epoch_{epoch+1}_{class_name}_ORIGINAL.ply"
    path_recon = (
        f"reconstructions_sanity/epoch_{epoch+1}_{class_name}_RECONSTRUCTED.ply"
    )

    o3d.io.write_point_cloud(path_orig, pcd_orig)
    o3d.io.write_point_cloud(path_recon, pcd_recon)
    print(f"\n[VISUAL] Amostra salva em {path_orig} e {path_recon}")


def run_sanity_check(
    model: DualBranchPointVAE,
    dataset,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Criamos o dataloader apenas para puxar o primeiro lote
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n" + "=" * 60)
    print("[INFO] Coletando 1 único batch para o Overfitting de Sanidade...")
    fixed_batch = next(iter(dataloader))
    class_names, pcd_data = fixed_batch

    # Trava os dados na GPU de forma estática
    xyz = pcd_data.to(device)
    target_class = class_names[0]
    print(
        f"[INFO] Batch coletado com sucesso! Classe da primeira amostra: '{target_class}'"
    )
    print("=" * 60 + "\n")

    # Usando AdamW que lida melhor com a estabilização de Transformers
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    # O loop agora itera diretamente sobre as épocas, já que o lote é fixo
    loop = tqdm(range(num_epochs), desc="Sanity Overfit")

    for epoch in loop:
        model.train()
        optimizer.zero_grad()

        # Forward pass no lote estático
        out = model(xyz)

        # Cálculo da Loss interna (Chamfer + KL)
        loss_dict = model.loss(out, xyz)

        loss_dict["total"].backward()
        optimizer.step()

        # Atualiza a barra de progresso instantaneamente
        loop.set_postfix(
            loss=f"{loss_dict['total'].item():.4f}",
            cd=f"{loss_dict['cd'].item():.4f}",
            kl=f"{loss_dict['kl'].item():.4f}",
        )

        # Salva amostras visuais no início, a cada 10 épocas e na última
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            save_before_after(xyz[0], out["recon"][0], epoch, f"sanity_{target_class}")

    print("\n" + "=" * 60)
    print("[SUCESSO] Teste de Sanidade Concluído!")
    print("[DICA] Verifique a pasta 'reconstructions_sanity' para avaliar o Overfit.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Dataset Sanity Check.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=int,
        default=False,
        help="1 if dataset is charged, 0 if not",
    )

    # Inicializa o modelo com os mesmos hiperparâmetros do treino real
    model = DualBranchPointVAE(
        d_model=384,
        latent_dim=512,
        n_out=2048,
        enc_depth=8,
        dec_depth=4,
        n_heads=6,
        beta=0,
    ).to(device)

    if parser.parse_args().dataset:
        print("[INFO] Carregando dataset pré-processado...")
        dataset = Ds_point_sampled_already()
    else:
        print("[INFO] Processando nuvens de pontos originais...")
        dataset = Ds_point_sampled(Ds_point_model())

    # Executa o teste controlado
    run_sanity_check(
        model=model,
        dataset=dataset,
        num_epochs=150,  # 50 épocas são mais que suficientes para esmagar 1 único batch
        batch_size=1,
        learning_rate=1e-3,
    )
