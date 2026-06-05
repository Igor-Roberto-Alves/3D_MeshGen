from src.Encoder import Encoder, EncoderStyle
from src.Decoder import LIONDecoder
# de src.metric import * (adicione se for usar as métricas aqui)
import torch
import torch.nn as nn

class Vae(nn.Module): 
    def __init__(self, latent_dim: int = 256, style_dim: int = 512, in_channels: int = 6):
        # 1. CORREÇÃO: Essencial para o PyTorch registrar os parâmetros do modelo
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # 2. CORREÇÃO: Instanciando as redes com seus respectivos hiperparâmetros
        self.style_encoder = EncoderStyle(in_channels=in_channels, style_dim=style_dim)
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=in_channels, style_dim=style_dim)
        
        # O LIONDecoder reconstrói coordenadas (3) + normais (3) = 6 canais de saída
        self.decoder = LIONDecoder(latent_dim=latent_dim, style_dim=style_dim, out_channels=6)
    
    def Encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sty = self.style_encoder(x) # Saída: (B, style_dim)
        
        mu, logvar = self.encoder(x, sty) # Saídas: (B, latent_dim)
        logvar = torch.clamp(logvar, min = -10.0, max = 10.0)
        # Precisamos retornar o 'sty' também para passá-lo depois para o Decoder!
        return mu, logvar, sty
    
    def Decode(self, z: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z     : (B, N, latent_dim) -> pontos latentes amostrados
        style : (B, style_dim)     -> estilo global
        """
        coords, normals = self.decoder(z, style)
        return coords, normals
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  
        mu, logvar, sty = self.Encode(x)
        
        # O reparameterization ocorre de forma única para cada ponto espacial da nuvem
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std  # Nuvem Latente Real de alta resolução: (B, N, latent_dim)
        
        # --- REMOVA OU COMENTE ESSAS LINHAS ARTIFICIAIS ---
        # N = x.shape[1]
        # z_points = z.unsqueeze(1).expand(-1, N, -1) 
        
        # Executa o decoder passando a textura latente rica e o estilo global
        coords_pred, normals_pred = self.Decode(z_points, sty)
        
        return coords_pred, normals_pred, mu, logvar
    
    def generate(self, num_samples: int, num_points: int = 2048, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gera nuvens de pontos puras amostrando diretamente do prior Gaussiano padrão N(0, I).
        
        num_samples : Quantidade de objetos 3D a gerar
        num_points  : Quantidade de pontos espaciais por objeto
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            # 1. Amostra a textura latente local por ponto do Prior Padrão
            z_points = torch.randn(num_samples, num_points, self.latent_dim, device=device)
            
            # 2. Amostra o vetor de estilo macro global do Prior Padrão
            sty_sample = torch.randn(num_samples, self.style_dim, device=device)
            
            # 3. Decodifica os vetores latentes em coordenadas e normais físicas
            coords_pred, normals_pred = self.Decode(z_points, sty_sample)
            
        return coords_pred, normals_pred