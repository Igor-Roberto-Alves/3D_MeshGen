import torch
from torch import nn
from src.Encoder import TransformerBlock


class PointDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 6,
        n_out: int = 2048,
    ):
        super().__init__()
        self.n_out = n_out
        self.mem_tokens = 64
        self.pts_per_token = n_out // self.mem_tokens  # 32

        # UPGRADE 1: Deep Non-Linear Inflation Layer
        self.z_to_mem = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, self.mem_tokens * d_model),
        )

        # UPGRADE 2: Context projection for the generation head
        self.z_to_context = nn.Sequential(nn.Linear(latent_dim, d_model), nn.GELU())

        self.query = nn.Parameter(torch.randn(1, self.mem_tokens, d_model) * 0.02)

        self.cross_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)

        # UPGRADE 3: Fixed Local 2D Surface Grid Prior (FoldingNet style)
        # Creates a static grid of coordinates ranging from -1 to 1
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 8), indexing="ij"
            ),
            dim=-1,
        ).reshape(self.pts_per_token, 2)
        self.register_buffer("local_grid", grid)

        # UPGRADE 4: High-Capacity Generation Head
        # Input size: d_model (token) + d_model (global z context) + 2 (grid coordinates)
        self.generation_head = nn.Sequential(
            nn.Linear(d_model + d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),  # Outputs 1 point (XYZ + Normal) at a time
        )

    def forward(self, z):
        B = z.shape[0]

        # 1. Unpack latent vector into memory sequence using deep capacity
        mem = self.z_to_mem(z).reshape(B, self.mem_tokens, d_model)
        global_context = self.z_to_context(z)  # (B, d_model)

        # 2. Transformer Processing
        q = self.query.expand(B, -1, -1)
        for blk in self.cross_blocks:
            q = blk(q, kv=mem)
        for blk in self.self_blocks:
            q = blk(q)
        q = self.norm(q)  # (B, 64, d_model)

        # 3. Spatial Expansion & Grid Injection
        # Expand tokens to map to every individual point query
        q_expanded = q.unsqueeze(2).expand(
            -1, -1, self.pts_per_token, -1
        )  # (B, 64, 32, d_model)

        # Inject the structural global context copy to every single point
        context_expanded = (
            global_context.unsqueeze(1)
            .unsqueeze(2)
            .expand(B, self.mem_tokens, self.pts_per_token, -1)
        )

        # Inject the local geometric grid template
        grid_expanded = self.local_grid.view(1, 1, self.pts_per_token, 2).expand(
            B, self.mem_tokens, -1, -1
        )

        # Concatenate everything together
        # Total channels: 384 (token) + 384 (global context) + 2 (grid) = 770
        combined_features = torch.cat(
            [q_expanded, context_expanded, grid_expanded], dim=-1
        )

        # 4. Dense Generation via High-Capacity MLP
        out = self.generation_head(combined_features)  # (B, 64, 32, 6)
        out = out.view(B, self.n_out, 6)

        # 5. Extract and normalize
        xyz = out[..., :3]
        normals = out[..., 3:]
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        return {"recon": torch.cat([xyz, normals], dim=-1)}
