import torch
from src.Vae import *
# Assumindo que o seu arquivo se chame modelo_vae.py e a classe seja Vae
# from modelo_vae import Vae 

if __name__ == "__main__":
    # 1. Configurações básicas e dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando o dispositivo: {device}\n")

    # 2. Inicializando o VAE (Configurado para 6 canais latentes e estilo 512)
    # latent_dim=3 + 3 canais de coordenadas físicas = 6 canais totais.
    vae = Vae(latent_dim=3, style_dim=512, in_channels=6).to(device)
    
    # Simulação de um lote (Batch) de 2 nuvens de pontos de entrada (ex: uma cadeira e uma mesa)
    # Formato: (Batch=2, Pontos=2048, Canais=6 -> XYZ + Normais)
    nuvem_entrada = torch.randn(1, 2048, 6).to(device)

    # =========================================================================
    # CENÁRIO A: UTILIZANDO O ENCODE E DECODE SEPARADAMENTE (Fluxo de Reconstrução)
    # =========================================================================
    print("=== [CENÁRIO A] Iniciando extração e reconstrução ===")
    
    vae.eval() # Modo de avaliação para travar camadas de Batch Normalization
    with torch.no_grad():
        # Passo 1: Passar a nuvem pelo Encode
        # mu e logvar representam os pontos latentes ricos (6 canais)
        # sty representa o vetor de estilo macro global (512x1)
        mu, logvar, sty = vae.Encode(nuvem_entrada)
        
        print(f"-> Sucesso no Encode!")
        print(f"   Shape do Vetor de Estilo Global (sty): {sty.shape} (Um vetor de 512 por objeto)")
        print(f"   Shape do mu da Nuvem Latente:        {mu.shape} (512 pontos com 6 canais cada)")
        print(f"   Shape do logvar da Nuvem Latente:    {logvar.shape}")

        # Passo 2: Amostragem no Gargalo (Reparameterization Trick manual)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std
        print(f"   Shape do z_points (Pronto para o Decoder): {z_points.shape}")

        # Passo 3: Passar os códigos latentes pelo Decode
        coords_reconstruidas, normals_reconstruidas = vae.Decode(z_points, sty)
        
        print(f"-> Sucesso no Decode!")
        print(f"   Nuvem Reconstruída - Coordenadas (XYZ): {coords_reconstruidas.shape}")
        print(f"   Nuvem Reconstruída - Normais:          {normals_reconstruidas.shape}")


    # =========================================================================
    # CENÁRIO B: UTILIZANDO O GENERATE (Criando novas formas do zero)
    # =========================================================================
    print("\n=== [CENÁRIO B] Gerando novas formas a partir do ruído ===")
    
    # Queremos gerar 4 objetos completamente inéditos
    num_novas_amostras = 4
    
    # O método generate faz o sorteio no espaço Gaussiano e chama o Decode internamente
    novas_coords, novas_normais = vae.generate(
        num_samples=num_novas_amostras, 
        num_points=512, 
        device=device
    )
    
    print(f"-> Sucesso no Generate!")
    print(f"   Objetos gerados inéditos: {novas_coords.shape[0]}")
    print(f"   Shape das Coordenadas Geradas (XYZ): {novas_coords.shape}")
    print(f"   Shape das Normais Geradas:          {novas_normais.shape}")

    # Próximo passo ideal no seu projeto:
    # salvar essas 'novas_coords' em um arquivo .ply ou .obj para visualizar no Blender/MeshLab!