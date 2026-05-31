import torch
from torch import nn
from src.Encoder import TransformerBlock

class PointDecoder(nn.Module):
    """
    Transformer-based point cloud generator.
    Learned query tokens attend to the latent code z to produce N × 6 points/features.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 4,
        n_out: int = 2048,
    ):
        super().__init__()
        self.n_out = n_out
        # Project z to d_model memory
        self.z_proj = nn.Linear(latent_dim, d_model)
        # Learned query tokens (one per output point — grouped by centres)
        self.n_centers = n_out // 32  # e.g. 64 centres × 32 pts/centre
        self.query = nn.Parameter(torch.randn(1, self.n_centers, d_model) * 0.02)
        # Cross-attention layers: queries attend to z
        self.cross_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        # Self-attention refinement
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)
        # Per-token MLP to predict 32 × 6 point coordinates + features
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 32 * 6),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, latent_dim)  →  (B, n_out, 6)"""
        B = z.shape[0]
        mem = self.z_proj(z).unsqueeze(1)  # (B, 1, d_model)
        q = self.query.expand(B, -1, -1)  # (B, n_centers, d_model)
        for blk in self.cross_blocks:
            q = blk(q, kv=mem)
        for blk in self.self_blocks:
            q = blk(q)
        q = self.norm(q)  # (B, n_centers, d_model)
        pts = self.head(q)  # (B, n_centers, 32*6)
        
        # FIX 1: Change 3 to 6 to handle the extra channels
        pts = pts.reshape(B, self.n_centers, 32, 6)
        

        spatial = pts[..., :3]
        features = pts[..., 3:]
        
        # Normalise ONLY spatial coordinates to a unit sphere
        spatial = spatial - spatial.mean(dim=(1, 2), keepdim=True)
        spatial = spatial / (spatial.norm(dim=-1, keepdim=True).max() + 1e-8)
        
        # Recombine them back into a single 6D tensor
        pts = torch.cat([spatial, features], dim=-1)
        
        return pts.reshape(B, self.n_out, 6)