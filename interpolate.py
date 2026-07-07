import torch
from torch.utils.tensorboard import SummaryWriter
from src.Vae import Vae
import argparse
from src.dataset import Ds_point_sampled_already


def normalise_batch(points: torch.Tensor) -> torch.Tensor:
    xyz    = points[..., :3]
    centre = xyz.mean(dim=1, keepdim=True)
    xyz    = xyz - centre
    scale  = xyz.norm(dim=2, keepdim=True).max(dim=1, keepdim=True).values.clamp(min=1e-6)
    xyz    = xyz / scale
    return torch.cat([xyz, points[..., 3:]], dim=2)

def main():
    # Argparser
    parser = argparse.ArgumentParser(description="Interpolação Latente Idêntica ao Treino")
    parser.add_argument('--weight', type=str, default='latest', help='Número da época (ex: "0229") ou "latest"')
    args = parser.parse_args()


    SEED = 42 
    torch.manual_seed(SEED)
    
    base_ds = Ds_point_sampled_already(root="point_clouds", augment=False)
    

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(base_ds), generator=generator).tolist()
    
    val_split = 0.1  
    val_n = max(1, int(len(base_ds) * val_split))
    val_idx = indices[:val_n]

    val_ds = torch.utils.data.Subset(
        Ds_point_sampled_already(root="point_clouds", augment=False),
        val_idx
    )

    writer = SummaryWriter(log_dir="latent_interpolation")
    
    latent_dim = 3
    style_dim = 512
    in_channels = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vae(latent_dim=latent_dim, style_dim=style_dim, in_channels=in_channels).to(device)

    if args.weight != "latest":
        weight_path = f"checkpoints/epoch_{args.weight}.pt"
    else:
        weight_path = "checkpoints/latest.pt"
    
    print(f"Carregando pesos de: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()  


    num_steps = 100
    num_exemplos = min(30, len(val_ds))  

    pair_generator = torch.Generator().manual_seed(1337)
    idx1_list = torch.randint(0, len(val_ds), (num_exemplos,), generator=pair_generator).tolist()
    idx2_list = torch.randint(0, len(val_ds), (num_exemplos,), generator=pair_generator).tolist()
    
    print(f"Gerando {num_exemplos} exemplos estruturados Lado a Lado de forma reproduzível...")
    
    with torch.no_grad():
        for ex in range(num_exemplos):
            idx1 = idx1_list[ex]
            idx2 = idx2_list[ex]
            
 
            if idx1 == idx2:
                idx2 = (idx2 + 1) % len(val_ds)
                
            sample1 = val_ds[idx1]
            sample2 = val_ds[idx2]
            
          
            img1 = sample1.unsqueeze(0).to(device)
            img2 = sample2.unsqueeze(0).to(device)


            img1 = normalise_batch(img1)
            img2 = normalise_batch(img2)

  
            mu1, logvar1, style1 = model.Encode(img1)
            mu2, logvar2, style2 = model.Encode(img2)
            
            z1 = model.z(mu1, logvar1, style1)
            z2 = model.z(mu2, logvar2, style2)

            rec1 = model(img1)[0]
            v1 = rec1.squeeze(0).cpu()
            v1_mesh = v1 if v1.shape[1] == 3 else v1.T
            num_pts1 = v1_mesh.shape[0]
            cor_azul = torch.tensor([0, 0, 255]).repeat(num_pts1, 1).unsqueeze(0)
            

            rec2 = model(img2)[0]
            v2 = rec2.squeeze(0).cpu()
            v2_mesh = v2 if v2.shape[1] == 3 else v2.T
            num_pts2 = v2_mesh.shape[0]
            cor_verde = torch.tensor([0, 255, 0]).repeat(num_pts2, 1).unsqueeze(0)

            writer.add_mesh(f"Caso_{ex:02d}/1_Dado_Original_1", vertices=v1_mesh.unsqueeze(0), colors=cor_azul, global_step=0)
            writer.add_mesh(f"Caso_{ex:02d}/3_Dado_Original_2", vertices=v2_mesh.unsqueeze(0), colors=cor_verde, global_step=0)

            for step in range(1, num_steps + 1):
                alpha = step / (num_steps + 1)
                
                # 1. Interpolação linear das médias e das log-variâncias
                mu_interp = (1.0 - alpha) * mu1 + alpha * mu2
                logvar_interp = (1.0 - alpha) * logvar1 + alpha * logvar2
                style_interp = (1.0 - alpha) * style1 + alpha * style2
                
                # 2. Aplica o Reparameterization Trick exatamente como no forward do VAE
                std = torch.exp(0.5 * logvar_interp)
                eps = torch.randn_like(std)
                z_interp_noisy = mu_interp + eps * std  # Mantém os 259 canais ricos e saudáveis
                
                # 3. O Decoder agora recebe o vetor latente com a mesma distribuição que foi treinado
                coords_pred, _ = model.Decode(z_interp_noisy, style_interp) 
                
                output_clean = coords_pred.squeeze(0).cpu()
                v_interp_mesh = output_clean if output_clean.shape[1] == 3 else output_clean.T
                
                num_pts_interp = v_interp_mesh.shape[0]
                cor_laranja = torch.tensor([255, 127, 0]).repeat(num_pts_interp, 1).unsqueeze(0)
                
                writer.add_mesh(
                    f"Caso_{ex:02d}/2_Interpolado", 
                    vertices=v_interp_mesh.unsqueeze(0), 
                    colors=cor_laranja,
                    global_step=step
                )
    writer.close()
    print(f"\nConcluído! Exemplos consistentes salvos em 'latent_interpolation'.")

if __name__ == "__main__":
    main()