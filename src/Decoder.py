import torch
from torch import nn
from src.Encoder import TransformerBlock

class PointDecoder(nn.Module):
    """
    Decoder baseado em Transformer para geração de nuvens de pontos.
    Gera centros globais e deslocamentos locais para garantir topologia perfeita.
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
        self.n_centers = n_out // 32  # 64 centros
        self.n_local_pts = 32         # 32 pontos por centro
        
        # Projetar z em 8 tokens de memória
        self.mem_tokens = 8
        self.z_proj = nn.Linear(latent_dim, self.mem_tokens * d_model)
        
        # Query tokens aprendidos (Inicialização ligeiramente maior para quebrar a simetria)
        self.query = nn.Parameter(torch.randn(1, self.n_centers, d_model) * 0.1)
        self.query = nn.Parameter(torch.randn(1, self.n_centers, d_model) * 0.2)
        # Camadas de Atenção
        self.cross_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(2)]
        )
        self.norm = nn.LayerNorm(d_model)
        
        # 🏢 Cabeça 1: Preve a posição ABSOLUTA global dos 64 centros
        self.center_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3), # Cospe X, Y, Z do centro
        )
        
        # 🛠️ Cabeça 2: Preve os deslocamentos locais e as normais para os 32 pontos
        self.local_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_local_pts * 6), # 32 vizinhos x (dx, dy, dz, nx, ny, nz)
        )
        
        self.spatial_act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> dict:
        """z : (B, latent_dim)  →  (B, n_out, 6)"""
        B = z.shape[0]
        
        # Criar a memória a partir do vetor latente
        mem = self.z_proj(z).reshape(B, self.mem_tokens, -1)
        q = self.query.expand(B, -1, -1)  # (B, n_centers, d_model)
        
        # Blocos de Atenção (Cross e Self)
        for blk in self.cross_blocks:
            q = blk(q, kv=mem)
        for blk in self.self_blocks:
            q = blk(q)
            
        q = self.norm(q)  # (B, n_centers, d_model)
        
        # 1. Prever os centros globais e aplicar Tanh para contê-los na esfera unitária
        centers_xyz = self.spatial_act(self.center_head(q)) # (B, n_centers, 3)
        centers_xyz = self.center_head(q)
        # 2. Prever as sub-estruturas locais
        local_features = self.local_head(q).reshape(B, self.n_centers, self.n_local_pts, 6)
        
        local_delta_xyz = self.spatial_act(local_features[..., :3]) # Deslocamentos locais controlados
        local_normals   = local_features[..., 3:]                    # Normais estimadas
        
        # Forçar as normais a serem unitárias para evitar explosão na loss de cosseno
        local_normals = local_normals / (local_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 3. Matemática Âncora: Ponto_Absoluto = Centro_Global + Deslocamento_Local
        # Usamos o unsqueeze(2) para somar o centro (64, 1, 3) aos 32 pontos locais (64, 32, 3)
        absolute_xyz = centers_xyz.unsqueeze(2) + (local_delta_xyz * 0.2) # Multiplicador 0.2 limita o tamanho do patch
        absolute_xyz = centers_xyz.unsqueeze(2) + (local_delta_xyz * 0.5)
        # Garantir que mesmo após a soma, nenhum ponto saia dos limites do Tanh do dataset
        absolute_xyz = torch.clamp(absolute_xyz, -1.0, 1.0)
        
        # Concatenar geometria absoluta com as normais unitárias
        pts = torch.cat([absolute_xyz, local_normals], dim=-1)
        
        # Redimensiona de volta para a nuvem linear de 2048 pontos
        recon = pts.reshape(B, self.n_out, 6)
        
        # Retornamos um dicionário para manter a compatibilidade com o teu loop de treino
        return {"recon": recon, "centers": centers_xyz}