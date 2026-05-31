from torch import nn
import torch
from typing import Tuple, Optional
from src.Tokenizer import PatchEmbed, PositionalEncoding, farthest_point_sampling, knn_group


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kv = x if kv is None else kv
        attn_out, _ = self.attn(self.norm1(x), self.norm1(kv), self.norm1(kv))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

# Blue Branch

class HierarchicalEncoder(nn.Module):

    """
    Three FPS + kNN grouping levels, each producing tokens that are
    concatenated and projected into d_model.
    """

    LEVELS = [
        dict(n_samples=512, k=32),
        dict(n_samples=256, k=32),
        dict(n_samples=128, k=32),
    ]

    def __init__(self, d_model: int = 384, n_heads: int = 6, depth: int = 4):
        super().__init__()
        self.d_model = d_model


        self.embeds = nn.ModuleList(
            [PatchEmbed(in_ch=6, out_ch=d_model, k=lv["k"]) for lv in self.LEVELS]
        )

        self.pos_encs = nn.ModuleList(
            [PositionalEncoding(d_model) for _ in self.LEVELS]
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )

        self.proj = nn.Linear(d_model * len(self.LEVELS), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        all_tokens, all_centers = [], []
        for i, lv in enumerate(self.LEVELS):
            idx = farthest_point_sampling(xyz, lv["n_samples"])
            centers = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.shape[-1]))   
            grouped, _ = knn_group(xyz, centers, lv["k"])
            

            tokens = self.embeds[i](grouped) + self.pos_encs[i](centers[..., :3])
            
            all_tokens.append(tokens)
            all_centers.append(centers)

        encoded_levels = []
        for i in range(len(self.LEVELS)):
            t = all_tokens[i]
            for blk in self.blocks:
                t = blk(t)
            encoded_levels.append(t.mean(dim=1))  # (B, d_model)

        centers = torch.cat(all_centers, dim=1)  # (B, M_total, 6)
        feat = torch.cat(encoded_levels, dim=-1)  # (B, d_model * 3)
        feat = self.norm(self.proj(feat))  # (B, d_model)
        return feat, centers


# green branch

class GlobalEncoder(nn.Module):
    """
    Single-level FPS grouping + transformer.  Captures global shape context.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 4,
        n_tokens: int = 64,
        k: int = 32,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed = PatchEmbed(in_ch=6, out_ch=d_model, k=k)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz : (B, N, 6)  →  (B, d_model)"""
        idx = farthest_point_sampling(xyz, self.n_tokens)
        centers = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.shape[-1]))   
        grouped, _ = knn_group(xyz, centers, k=32)
        

        tokens = self.embed(grouped) + self.pos_enc(centers[..., :3])
        
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens.mean(dim=1))

# Attention fusion


class CrossBranchFusion(nn.Module):
    """
    Bidirectional cross-attention between hierarchical and global feature vectors.
    Both are unsqueezed to sequence length 1 for nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attn_h2g = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_g2h = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """h, g : (B, d_model)  →  fused (B, d_model)"""
        h_s = h.unsqueeze(1)  # (B, 1, d_model)
        g_s = g.unsqueeze(1)
        h2g, _ = self.attn_h2g(self.norm_h(h_s), self.norm_g(g_s), self.norm_g(g_s))
        g2h, _ = self.attn_g2h(self.norm_g(g_s), self.norm_h(h_s), self.norm_h(h_s))
        h_out = (h_s + h2g).squeeze(1)
        g_out = (g_s + g2h).squeeze(1)
        return self.proj(torch.cat([h_out, g_out], dim=-1))




class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparametrisation
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(μ,σ²) ‖ N(0,1) )  averaged over the batch."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())