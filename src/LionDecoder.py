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


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.norm1 = nn.GroupNorm(8, channels)

        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):

        res = x

        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        return F.gelu(x + res)


class StablePointDecoder(nn.Module):

    """
    Lightweight reconstruction decoder.

    ~5M params instead of hundreds of millions.
    """

    def __init__(
        self,
        latent_dim=512,
        style_dim=384,
        n_out=2048,
        n_seed=256,
        hidden=128,
    ):
        super().__init__()

        self.n_out = n_out
        self.n_seed = n_seed
        self.expand_ratio = n_out // n_seed
        self.hidden = hidden

        # ------------------------------------------------------------
        # Small latent expansion
        # ------------------------------------------------------------

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + style_dim, 1024),
            nn.GELU(),

            nn.Linear(1024, n_seed * hidden),
            nn.GELU(),
        )

        # ------------------------------------------------------------
        # Learned coarse seed positions
        # ------------------------------------------------------------

        self.seed_xyz = nn.Parameter(
            torch.randn(1, 3, n_seed) * 0.02
        )

        # ------------------------------------------------------------
        # Geometry refinement
        # ------------------------------------------------------------

        self.refine = nn.Sequential(
            ResidualBlock1D(hidden),
            ResidualBlock1D(hidden),
            ResidualBlock1D(hidden),
        )

        # ------------------------------------------------------------
        # Upsampling features
        # ------------------------------------------------------------

        self.upsample = nn.Conv1d(
            hidden,
            hidden * self.expand_ratio,
            kernel_size=1
        )

        # ------------------------------------------------------------
        # XYZ prediction
        # ------------------------------------------------------------

        self.xyz_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),

            nn.Conv1d(hidden, 3, 1),
        )

        # ------------------------------------------------------------
        # Normal prediction
        # ------------------------------------------------------------

        self.normal_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),

            nn.Conv1d(hidden, 3, 1),
        )

    def forward(self, z_local, style):

        B = z_local.shape[0]

        # ------------------------------------------------------------
        # Merge latent + style
        # ------------------------------------------------------------

        latent = torch.cat(
            [z_local, style],
            dim=-1
        )

        # ------------------------------------------------------------
        # Create coarse features
        # ------------------------------------------------------------

        feat = self.fc(latent)

        feat = feat.view(
            B,
            self.n_seed,
            self.hidden
        )

        feat = feat.transpose(1, 2)

        # ------------------------------------------------------------
        # Refine coarse geometry
        # ------------------------------------------------------------

        feat = self.refine(feat)

        # ------------------------------------------------------------
        # Upsample point features
        # ------------------------------------------------------------

        feat = self.upsample(feat)

        feat = feat.view(
            B,
            self.hidden,
            self.expand_ratio,
            self.n_seed
        )

        feat = feat.permute(0, 1, 3, 2)

        feat = feat.reshape(
            B,
            self.hidden,
            self.n_out
        )

        # ------------------------------------------------------------
        # Predict local offsets
        # ------------------------------------------------------------

        xyz_offset = self.xyz_head(feat)

        normals = self.normal_head(feat)

        normals = F.normalize(
            normals,
            dim=1
        )

        # ------------------------------------------------------------
        # Expand coarse anchors
        # ------------------------------------------------------------

        anchors = self.seed_xyz.expand(
            B,
            -1,
            -1
        )

        anchors = anchors.unsqueeze(-1)

        anchors = anchors.expand(
            -1,
            -1,
            -1,
            self.expand_ratio
        )

        anchors = anchors.reshape(
            B,
            3,
            self.n_out
        )

        xyz = anchors + xyz_offset

        # ------------------------------------------------------------
        # Output format
        # ------------------------------------------------------------
        xyz = xyz.clamp(-1.0, 1.0)
        xyz = xyz.transpose(1, 2)

        normals = normals.transpose(1, 2)

        recon = torch.cat(
            [xyz, normals],
            dim=-1
        )

        return {
            "recon": recon
        }
