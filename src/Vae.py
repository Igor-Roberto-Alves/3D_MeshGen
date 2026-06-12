from src.Encoder import Encoder, EncoderStyle
from src.Decoder import LIONDecoder
import torch
import torch.nn as nn

class Vae(nn.Module): 
    def __init__(self, latent_dim: int = 256, style_dim: int = 512, in_channels: int = 6):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        # Branchs
        self.style_encoder = EncoderStyle(in_channels=in_channels, style_dim=style_dim)
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=in_channels, style_dim=style_dim)
        

        self.decoder = LIONDecoder(latent_dim=latent_dim, style_dim=style_dim, input_dim=3)
    
    def Encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_style, logvar_style = self.style_encoder(x) # Saída: (B, style_dim)
        sty =  self.style_encoder.z(mu_style, logvar_style)
        mu, logvar = self.encoder(x, sty) 
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar, sty, mu_style, logvar_style
    
    def Decode(self, z: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z     : (B, N, latent_dim + input_dim) -> pontos latentes (Ex: 259 canais)
        style : (B, style_dim)                 -> estilo global
        """
        coords, normals = self.decoder(z, style)
        return coords, normals
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, sty, mu_style, logvar_style = self.Encode(x)
        
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std  
        

        coords_pred, normals_pred = self.Decode(z_points, sty)
        
        return coords_pred, normals_pred, mu, logvar, mu_style, logvar_style
    
    def z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_points = mu + eps * std  # Nuvem Latente: (B, 128, 259)
        return z_points

    def generate(self, num_samples: int, num_points: int = 512, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
 
        # Samples of N(0, I)
        
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():

            sty_sample = torch.randn(num_samples, self.style_dim, device=device)
            
        
            feat_sample = torch.randn(num_samples, num_points, self.latent_dim, device=device)

            xyz_anchors = torch.randn(num_samples, num_points, 3, device=device)
            xyz_anchors = torch.nn.functional.normalize(xyz_anchors, p=2, dim=-1)
            
 
            r = torch.rand(num_samples, num_points, 1, device=device) ** (1/3)
            xyz_anchors = xyz_anchors * r 
            
   
            z_points = torch.cat([xyz_anchors, feat_sample], dim=-1) 
            
        
            coords_pred, normals_pred = self.Decode(z_points, sty_sample)
            
        return coords_pred, normals_pred