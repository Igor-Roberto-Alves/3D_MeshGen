import torch
from torch import nn
from src.Encoder import TransformerBlock

class PointDecoder(nn.Module):
    def __init__(self,
        latent_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 6,
        n_out: int = 2048,
    ):
        super().__init__()
        self.n_out = n_out
        self.mem_tokens = 64
        self.pts_per_token = n_out // self.mem_tokens # 2048 / 64 = 32
        
        self.z_proj = nn.Linear(latent_dim, self.mem_tokens * d_model)
        self.query = nn.Parameter(torch.randn(1, self.mem_tokens, d_model) * 0.02)
        
        self.cross_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)

        # APENAS a nova cabeça de geração livre
        self.generation_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.pts_per_token * 6) 
        )

    # O seu forward() está perfeito!
    def forward(self, z):
        B = z.shape[0]
        
        mem = self.z_proj(z).reshape(B, self.mem_tokens, -1)
        q = self.query.expand(B, -1, -1)   

        for blk in self.cross_blocks:
            q = blk(q, kv=mem)
        for blk in self.self_blocks:
            q = blk(q)
            
        q = self.norm(q)   
        out = self.generation_head(q) 
        
        out = out.view(B, self.n_out, 6)
        
        xyz = out[..., :3]
        normals = out[..., 3:]
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        return {"recon": torch.cat([xyz, normals], dim=-1)}