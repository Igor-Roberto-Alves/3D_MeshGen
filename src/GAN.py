import torch
import torch.nn as nn


class PointNetDiscriminator(nn.Module):
    """
    PointNet-style discriminator for point clouds (B, N, 6).

    Processes each point independently via a shared MLP, aggregates with
    global max-pooling, then classifies real / fake with a final MLP.

    Args:
        in_ch        : input channels per point (6 = XYZ + normals)
        base_ch      : base channel width for internal layers
        use_spectral : apply spectral norm to every Linear layer
                       (stabilises GAN training without batch norm)
    """

    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = 64,
        use_spectral: bool = True,
    ):
        super().__init__()

        def linear(a: int, b: int) -> nn.Module:
            layer = nn.Linear(a, b)
            return nn.utils.spectral_norm(layer) if use_spectral else layer

        # Shared point-wise MLP: (B, N, in_ch) → (B, N, base_ch*8)
        self.mlp = nn.Sequential(
            linear(in_ch,          base_ch),        #   6 → 64
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch,        base_ch * 2),    #  64 → 128
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 2,    base_ch * 4),    # 128 → 256
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 4,    base_ch * 8),    # 256 → 512
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global classifier: (B, base_ch*8) → (B, 1)
        self.head = nn.Sequential(
            linear(base_ch * 8,    base_ch * 4),    # 512 → 256
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch * 4,    base_ch),         # 256 → 64
            nn.LeakyReLU(0.2, inplace=True),
            linear(base_ch,        1),               #  64 → 1 (logit)
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts    : (B, N, 6)
        returns: logits (B, 1) — no sigmoid; use BCEWithLogitsLoss
        """
        feat = self.mlp(pts)               # (B, N, base_ch*8)
        feat = feat.max(dim=1).values      # (B, base_ch*8)  global max-pool
        return self.head(feat)             # (B, 1)