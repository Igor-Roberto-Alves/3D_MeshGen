import torch
import torch.nn as nn



def normalize_pc(x: torch.Tensor) -> torch.Tensor:

    xyz  = x[..., :3]
    rest = x[..., 3:]
    centre = xyz.mean(dim=1, keepdim=True)                                 # (B, 1, 3)
    xyz_c  = xyz - centre

    scale  = (xyz_c.abs()
               .max(dim=-1, keepdim=True).values   # per-point max over xyz
               .max(dim=1,  keepdim=True).values    # global max over points
               .clamp(min=1e-6))                    # (B, 1, 1)
    return torch.cat([xyz_c / scale, rest], dim=-1)

