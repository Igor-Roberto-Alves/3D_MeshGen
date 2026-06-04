import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class StyleModulationBlock(nn.Module):
    """Modula as características dos pontos locais usando o estilo global (estilo AdaGN)."""
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.fc = nn.Sequential(
            nn.Linear(style_dim, channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, style):
        # x: (B, channels, N)
        # style: (B, style_dim)
        norm_x = self.norm(x)
        style_embs = self.fc(style).unsqueeze(-1) # (B, channels * 2, 1)
        gamma, beta = torch.chunk(style_embs, 2, dim=1)
        
        out = norm_x * (1 + gamma) + beta
        return out + self.conv_res(x)


class LatentPointAttention(nn.Module):
    """Self-Attention para fazer os pontos latentes vizinhos cooperarem geometricamente."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        B, C, N = x_norm.shape
        
        qkv = self.to_qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        q = q.view(B, self.heads, C // self.heads, N)
        k = k.view(B, self.heads, C // self.heads, N)
        v = v.view(B, self.heads, C // self.heads, N)
        
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(B, C, N)
        return res + self.to_out(out)


import torch
import torch.nn as nn
import torch.nn.functional as F

class NvidiaStyleLatentDecoder(nn.Module):
    """
    Decoder robusto baseado no modelo da NVIDIA.
    Compatível com o Bottleneck flat de 512 dimensões.
    """
    def __init__(self, latent_dim=512, style_dim=384, n_out=2048, n_seed=256):
        super().__init__()
        self.n_out = n_out
        self.n_seed = n_seed
        self.expand_ratio = n_out // n_seed  # Ex: 8
        self.feature_dim = 256  # Tamanho fixo e estável para as features dos pontos
        
        # ── EXPANSÃO DO VETOR FLAT ──
        # Transforma o vetor z de 512 em uma matriz rica de (n_seed * feature_dim)
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, n_seed * self.feature_dim),
            nn.GELU()
        )
        
        # Âncoras espaciais aprendidas (como no seu LION original)
        self.seed_points = nn.Parameter(torch.randn(1, 3, n_seed) * 0.1)

        # Camadas de processamento e modulação por estilo
        self.layer1_conv = nn.Conv1d(self.feature_dim, 256, kernel_size=1)
        self.layer1_mod  = StyleModulationBlock(256, style_dim)
        
        self.layer2_conv = nn.Conv1d(256, 512, kernel_size=1)
        self.layer2_mod  = StyleModulationBlock(512, style_dim)
        
        self.attention   = LatentPointAttention(512, heads=8)
        
        self.layer3_conv = nn.Conv1d(512, 1024, kernel_size=1)
        self.layer3_mod  = StyleModulationBlock(1024, style_dim)

        # Cabeças de predição locais (Deltas de deslocamento e Normais)
        self.xyz_delta = nn.Conv1d(1024, self.expand_ratio * 3, kernel_size=1)
        self.normal_head = nn.Conv1d(1024, self.expand_ratio * 3, kernel_size=1)

    def forward(self, z_local, style):
        # z_local: (B, 512)
        # style: (B, 384) -> vindo diretamente do seu glob_enc
        B = z_local.shape[0]
        
        # 1. Expandir o vetor latente plano para a grade de sementes (seed grid)
        # (B, 512) -> (B, 256 * 256) -> (B, 256, 256) -> Transpose para (B, 256, 256) canais/pontos
        latent_feats = self.latent_expand(z_local).view(B, self.n_seed, self.feature_dim)
        latent_feats = latent_feats.transpose(1, 2) # (B, 256, 256)
        
        # Pegar as âncoras espaciais aprendidas e expandir para o tamanho do Batch
        anchor_xyz = self.seed_points.expand(B, -1, -1) # (B, 3, 256)
        
        # 2. Fluxo de extração guiado pelo estilo
        feat = F.gelu(self.layer1_conv(latent_feats))
        feat = self.layer1_mod(feat, style)
        
        feat = F.gelu(self.layer2_conv(feat))
        feat = self.layer2_mod(feat, style)
        
        # Comunicação por Atenção das features globais
        feat = self.attention(feat)
        
        feat = F.gelu(self.layer3_conv(feat))
        feat = self.layer3_mod(feat, style)
        
        # 3. Geração dos Deslocamentos (Deltas) e Normais
        deltas = self.xyz_delta(feat).view(B, self.expand_ratio, 3, self.n_seed)
        normals = self.normal_head(feat).view(B, self.expand_ratio, 3, self.n_seed)
        
        # 4. Deformação Local (Ancoragem Espacial)
        anchors_expanded = anchor_xyz.unsqueeze(1).expand(-1, self.expand_ratio, -1, -1)
        final_xyz = anchors_expanded + deltas 
        
        # Ajuste final para o formato de saída [B, N_out, 3]
        final_xyz = final_xyz.permute(0, 3, 1, 2).reshape(B, self.n_out, 3)
        final_normals = normals.permute(0, 3, 1, 2).reshape(B, self.n_out, 3)
        
        final_normals = final_normals / (final_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        return {"recon": torch.cat([final_xyz, final_normals], dim=-1)}