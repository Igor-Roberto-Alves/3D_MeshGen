import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """Point-wise residual block (B, C, N)."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FoldingLayer(nn.Module):
    """
    Single folding step: maps (z_exp || grid || prev) → new 3-D coords.

    in_channels  : z_dim + grid_dim + prev_coord_dim
    hidden       : hidden layer width
    """
    def __init__(self, in_channels: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden,  1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden,      hidden,  1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden,      3,       1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    FoldingNet-style decoder conditioned on a latent vector z.

    Pipeline
    --------
    1. Expand z to every point: (B, latent_dim) → (B, latent_dim, N)
    2. Build a 2-D canonical grid of N points (flattened √N × √N).
    3. Two folding steps produce coarse 3-D coordinates.
    4. A refinement MLP predicts per-point residuals (normal-like offsets).
    5. Output: coarse point cloud  (B, N, 3)
               refined point cloud (B, N, 3)   ← primary output
               per-point normals   (B, N, 3)

    Args
    ----
    latent_dim    : dimension of the input latent code z
    num_points    : number of output points (must be a perfect square)
    hidden        : hidden MLP width in folding layers
    """

   
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        hidden: int = 512,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_points = num_points

        # ------------------------------------------------------------------
        # Flexible 2-D grid (works for ANY num_points)
        # ------------------------------------------------------------------

        h = int(num_points ** 0.5)
        w = (num_points + h - 1) // h

        self.grid_h = h
        self.grid_w = w

        # canonical 2-D grid
        u = torch.linspace(-1, 1, h)
        v = torch.linspace(-1, 1, w)

        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")

        grid = torch.stack(
            [
                grid_u.flatten(),
                grid_v.flatten(),
            ],
            dim=0,
        )  # (2, h*w)

        # trim excess points
        grid = grid[:, :num_points]

        self.register_buffer("grid", grid)

        grid_dim = 2


        # ---- latent projection ----------------------------------------
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # ---- fold 1: (z || grid) → coarse coords ----------------------
        self.fold1 = FoldingLayer(hidden + grid_dim, hidden)

        # ---- fold 2: (z || fold1_out) → refined coarse coords ----------
        self.fold2 = FoldingLayer(hidden + 3, hidden)

        # ---- per-point feature refinement for residuals ----------------
        self.refine_backbone = nn.Sequential(
            nn.Conv1d(hidden + 3, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            ResidualMLP(hidden),
            ResidualMLP(hidden),
        )

        self.refine_xyz     = nn.Conv1d(hidden, 3, 1)   # residual Δxyz
        self.refine_normals = nn.Conv1d(hidden, 3, 1)   # predicted normals

    # ------------------------------------------------------------------
    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (B, latent_dim)

        Returns
        -------
        coarse   : (B, N, 3)  – after fold2
        refined  : (B, N, 3)  – coarse + residual
        normals  : (B, N, 3)  – unit normals (not normalised here; loss handles it)
        """
        B  = z.shape[0]
        N  = self.num_points

        # project latent → (B, hidden)
        z_feat = self.z_proj(z)                              # (B, hidden)

        # expand to all points → (B, hidden, N)
        z_exp  = z_feat.unsqueeze(2).expand(-1, -1, N)

        # canonical grid → (B, 2, N)
        grid   = self.grid.unsqueeze(0).expand(B, -1, -1)

        # ---- fold 1 ---------------------------------------------------
        inp1   = torch.cat([z_exp, grid], dim=1)            # (B, hidden+2, N)
        f1     = self.fold1(inp1)                            # (B, 3, N)

        # ---- fold 2 ---------------------------------------------------
        inp2   = torch.cat([z_exp, f1], dim=1)              # (B, hidden+3, N)
        f2     = self.fold2(inp2)                            # (B, 3, N)

        coarse = f2.permute(0, 2, 1)                        # (B, N, 3)

        # ---- refinement -----------------------------------------------
        ref_in  = torch.cat([z_exp, f2], dim=1)             # (B, hidden+3, N)
        ref_h   = self.refine_backbone(ref_in)               # (B, hidden, N)

        delta   = self.refine_xyz(ref_h)                    # (B, 3, N)
        normals = self.refine_normals(ref_h)                 # (B, 3, N)

        refined = (f2 + delta).permute(0, 2, 1)             # (B, N, 3)
        normals = normals.permute(0, 2, 1)                   # (B, N, 3)

        return coarse, refined, normals