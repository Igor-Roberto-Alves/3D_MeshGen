
import torch
import torch.nn as nn
import torch.nn.functional as F


class StablePointNetDiscriminator(nn.Module):
    """
    Simple and stable PointNet discriminator for point clouds.

    Input:
        (B, N, 6)

    Output:
        (B, 1) logits
    """

    def __init__(
        self,
        in_ch: int = 6,
        hidden: int = 128,
        use_spectral: bool = False,
    ):
        super().__init__()

        def linear(in_f, out_f):
            layer = nn.Linear(in_f, out_f)

            if use_spectral:
                layer = nn.utils.spectral_norm(layer)

            return layer

        self.mlp = nn.Sequential(
            linear(in_ch, hidden),
            nn.LeakyReLU(0.2, inplace=True),

            linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),

            linear(hidden, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),

            linear(hidden * 2, hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            linear(hidden * 4, hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),

            linear(hidden * 2, 1),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts:
            (B, N, 6)
        """

        feat = self.mlp(pts)

        # Global max pooling
        feat = feat.max(dim=1).values

        logits = self.head(feat)

        # safety clamp
        logits = torch.clamp(logits, -20, 20)

        return logits
