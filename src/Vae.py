from src.Encoder import Encoder, EncoderStyle
from src.Decoder import LIONDecoder
import torch
import torch.nn as nn

class Vae(nn.Module): 
    def __init__(self, latent_dim: int = 256, style_dim: int = 512, in_channels: int = 6):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # Instanciando as redes com seus respectivos hiperparâmetros
        self.style_encoder = EncoderStyle(in_channels=in_channels, style_dim=style_dim)
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=in_channels, style_dim=style_dim)
        
        # CORREÇÃO 1: Passar corretamente o input_dim=3 (XYZ) para que a soma interna dê 259 canais
        self.decoder = LIONDecoder(latent_dim=latent_dim, style_dim=style_dim, input_dim=3)
    
    def Encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sty = self.style_encoder(x) # Saída: (B, style_dim)
        
        mu, logvar = self.encoder(x, sty) # Saídas: (B, N, 3 + 256) por conta do truque do LION
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar, sty
    
    def Decode(self, z: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z     : (B, N, latent_dim + input_dim) -> pontos latentes (Ex: 259 canais)
        style : (B, style_dim)                 -> estilo global
        """
        coords, normals = self.decoder(z, style)
        return coords, normals
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, sty = self.Encode(x)
        
        # O reparameterization ocorre preservando os 259 canais gerados pelo Encoder
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std  # Nuvem Latente: (B, N, 259)
        
        # Executa o decoder passando os pontos latentes ricos e o estilo global
        coords_pred, normals_pred = self.Decode(z_points, sty)
        
        return coords_pred, normals_pred, mu, logvar
    
    def z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std  # Nuvem Latente: (B, N, 259)
        return z_points

    def generate(self, num_samples: int, num_points: int = 512, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gera nuvens de pontos puras amostrando diretamente do prior Gaussiano padrão N(0, I).
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            # CORREÇÃO 2: Na amostragem/inferência, você sorteia a "argila mágica" completa!
            # São 3 canais de coordenadas XYZ esféricas aleatórias + 256 canais de puro ruído abstrato.
            # Totalizando 259 canais por ponto. O número de pontos padrão do LION no gargalo é 512.
            total_channels = self.latent_dim + 3 # 256 + 3 = 259
            z_points = torch.randn(num_samples, num_points, total_channels, device=device)
            
            # Amostra o vetor de estilo macro global do Prior Padrão
            sty_sample = torch.randn(num_samples, self.style_dim, device=device)
            
            # Decodifica os vetores latentes em coordenadas e normais físicas
            coords_pred, normals_pred = self.Decode(z_points, sty_sample)
            
        return coords_pred, normals_pred