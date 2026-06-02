import os
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader
from src.Vae import DualBranchPointVAE
from src.dataset import Ds_point_sampled, Ds_point_model

def force_overfit():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SANITY] Usando dispositivo: {device}")

    print("[SANITY] Instanciando o modelo com beta=0.0...")
    model = DualBranchPointVAE(
        d_model   = 384,
        latent_dim= 1024*4,
        n_out     = 2048, # <-- Investigando o impacto disso
        enc_depth = 6,
        dec_depth = 6,
        n_heads   = 6,
        beta      = 0.0,  
    ).to(device)

    print("[SANITY] Carregando uma única amostra geométrica fixa...")
    base_dataset = Ds_point_model()
    dataset = Ds_point_sampled(base_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    class_name_list, fixed_pcd = next(iter(dataloader))
    class_name = class_name_list[0]
    
    xyz_fixed = fixed_pcd.to(device).clone().detach()
    
    print(f"[SANITY] Dado fixado com sucesso. Classe: {class_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    num_epochs = 7000
    print(f"\n[SANITY] Iniciando loop de overfit por {num_epochs} épocas...")
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        out = model(xyz_fixed)              
        
        # ─── BLOCO DE INSPEÇÃO GEOMÉTRICA (RODA APENAS NA ÉPOCA 0) ───
        if epoch == 0:
            print("\n" + "="*50)
            print("🕵️ ANÁLISE DE COMPATIBILIDADE DIMENSIONAL 🕵️")
            print("="*50)
            print(f"👉 DADO DE ENTRADA (xyz_fixed):")
            print(f"   - Shape: {xyz_fixed.shape} -> Esperado: [Batch, N_pontos, Canais_coordenadas_e_normais]")
            print(f"   - Tipo:  {xyz_fixed.dtype}")
            print(f"   - Device: {xyz_fixed.device}")
            print("-" * 50)
            
            # Investiga a estrutura do dicionário de saída
            print(f"👉 DADO DE SAÍDA DO MODELO (out):")
            print(f"   - Chaves geradas pelo modelo: {list(out.keys())}")
            
            if "recon" in out:
                recon_tensor = out["recon"]
                print(f"   - Shape de out['recon']: {recon_tensor.shape}")
                print(f"   - Tipo de out['recon']:  {recon_tensor.dtype}")
                print(f"   - Device de out['recon']: {recon_tensor.device}")
                
                # Verificação de compatibilidade direta
                if recon_tensor.shape == xyz_fixed.shape:
                    print("\n🟩 COMPATÍVEL: Os shapes de entrada e saída são idênticos!")
                else:
                    print("\n🟥 INCOMPATÍVEL: Os shapes de entrada e saída divergem!")
                    print(f"   Diferença gritante: Entrada {list(xyz_fixed.shape)} vs Saída {list(recon_tensor.shape)}")
                    print("   *Nota: Se a perda Chamfer não souber lidar com shapes diferentes internamente, ela falhará ou estagnará.*")
            else:
                print("🟥 ERRO CRÍTICO: A chave 'recon' não foi encontrada no output do modelo.")
            print("="*50 + "\n")
        # ─────────────────────────────────────────────────────────────

        loss_dict = model.loss(out, xyz_fixed)
        
        loss_dict["total"].backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Época [{epoch + 1:03d}/{num_epochs}] -> "
                f"Loss Total (CD): {loss_dict['total'].item():.6f} | "
                f"CD Pura: {loss_dict['cd'].item():.6f} | "
                f"KL (Ignorada): {loss_dict['kl'].item():.4f}"
            )

    print("\n[SANITY] Treinamento de sanidade concluído!")
    
    model.eval()
    with torch.no_grad():
        final_out = model(xyz_fixed)
        final_recon = final_out["recon"]

    os.makedirs("sanity_results", exist_ok=True)
    
    orig_np = xyz_fixed[0].cpu().numpy()
    recon_np = final_recon[0].cpu().numpy()

    # Tratamento dinâmico para evitar crashes caso as dimensões de saída estejam quebradas (ex: achatadas em 1D ou 2D)
    pcd_orig = o3d.geometry.PointCloud()
    if len(orig_np.shape) == 2 and orig_np.shape[1] >= 3:
        pcd_orig.points = o3d.utility.Vector3dVector(orig_np[:, :3])
        if orig_np.shape[1] >= 6:
            pcd_orig.normals = o3d.utility.Vector3dVector(orig_np[:, 3:])

    pcd_recon = o3d.geometry.PointCloud()
    if len(recon_np.shape) == 2 and recon_np.shape[1] >= 3:
        pcd_recon.points = o3d.utility.Vector3dVector(recon_np[:, :3])
        if recon_np.shape[1] >= 6:
            pcd_recon.normals = o3d.utility.Vector3dVector(recon_np[:, 3:])
    elif len(recon_np.shape) == 1:
         print(f"⚠️ [AVISO] O dado de saída está achatado (1D) com tamanho {recon_np.shape}. Tentando redimensionar para XYZ...")
         try:
             recon_np_reshaped = recon_np.reshape(-1, 6)
             pcd_recon.points = o3d.utility.Vector3dVector(recon_np_reshaped[:, :3])
             pcd_recon.normals = o3d.utility.Vector3dVector(recon_np_reshaped[:, 3:])
         except Exception as e:
             print(f"Não foi possível converter a saída para pontos 3D: {e}")

    o3d.io.write_point_cloud("sanity_results/SANITY_ORIGINAL.ply", pcd_orig)
    o3d.io.write_point_cloud("sanity_results/SANITY_OVERFITTED.ply", pcd_recon)
    
    print("[INFO] Arquivos de sanidade salvos na pasta 'sanity_results/'.")

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    force_overfit()