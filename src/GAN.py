"""
src/GAN.py  –  Improved Discriminator for DualBranchPointVAE
=============================================================

Architecture improvements over the baseline PointNet discriminator:

1.  **Multi-Scale Feature Extractor** — Three parallel MLP branches (narrow /
    medium / wide receptive fields via different channel widths) whose outputs
    are concatenated before global pooling.  Gives the discriminator richer
    inductive bias without increasing depth linearly.

2.  **STN-lite (mini T-Net)** — A 3-D spatial transformer network that aligns
    the input point cloud before feature extraction, making the discriminator
    rotation-invariant and harder to fool with trivial geometric tricks.

3.  **Spectral Norm on every Linear layer** (optional, on by default) — same
    as the original, keeps Lipschitz constant bounded.

4.  **Gradient Penalty helper** (``compute_gradient_penalty``) — drop-in
    support for WGAN-GP / R1 training; the train loop can call it when it
    wants a stronger regularisation signal.

5.  **Self-Attention bottleneck** — a single lightweight self-attention layer
    after global pooling captures global structure correlations that
    max-pooling alone discards.

6.  **Label-Smoothing-aware logit output** — still raw logits (no sigmoid),
    fully compatible with ``BCEWithLogitsLoss`` and WGAN critics alike.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sn_linear(in_f: int, out_f: int, spectral: bool = True) -> nn.Linear:
    layer = nn.Linear(in_f, out_f)
    return nn.utils.spectral_norm(layer) if spectral else layer


def _sn_conv1d(in_c: int, out_c: int, spectral: bool = True) -> nn.Conv1d:
    layer = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)
    return nn.utils.spectral_norm(layer) if spectral else layer


# ─────────────────────────────────────────────────────────────────────────────
# Mini T-Net  (spatial transformer for 3-D coords)
# ─────────────────────────────────────────────────────────────────────────────

class _MiniTNet(nn.Module):
    """
    Lightweight 3-D spatial transformer.
    Predicts a 3×3 rotation-ish matrix from the XYZ centroid features and
    applies it to the XYZ columns of the input point cloud.

    Keeping it small (two linear layers) so it doesn't dominate the param
    budget.  The output matrix is *regularised* toward the identity via the
    orthogonality loss ``tnet_reg_loss``.
    """

    def __init__(self, spectral: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            _sn_linear(3, 64, spectral),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_linear(64, 128, spectral),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            _sn_linear(128, 64, spectral),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_linear(64, 9, spectral),   # → 3×3 matrix
        )
        # Initialise last layer to near-identity
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.eye_(self.fc[-1].bias.view(3, 3))   # type: ignore[arg-type]

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        xyz : (B, N, 3)
        returns:
            aligned : (B, N, 3)
            mat     : (B, 3, 3)  — for optional reg loss
        """
        B = xyz.size(0)
        feat = self.mlp(xyz)                   # (B, N, 128)
        feat = feat.max(dim=1).values          # (B, 128)
        mat  = self.fc(feat).view(B, 3, 3)     # (B, 3, 3)
        aligned = torch.bmm(xyz, mat)          # (B, N, 3)
        return aligned, mat

    @staticmethod
    def tnet_reg_loss(mat: torch.Tensor) -> torch.Tensor:
        """
        Frobenius loss ||I - A·Aᵀ||² that penalises deviation from
        orthogonality.  Add a small weight (e.g. 1e-3) to the total loss.
        """
        B = mat.size(0)
        I = torch.eye(3, device=mat.device).unsqueeze(0).expand(B, -1, -1)
        diff = I - torch.bmm(mat, mat.transpose(1, 2))
        return (diff ** 2).sum(dim=(1, 2)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Self-Attention Bottleneck (single-head, lightweight)
# ─────────────────────────────────────────────────────────────────────────────

class _SelfAttention1D(nn.Module):
    """
    Single-head self-attention over a 1-D sequence (here the global feature
    vector is treated as a sequence of ``seq`` tokens of dimension ``d``).
    The same module also works point-wise when applied before pooling.
    """

    def __init__(self, dim: int, spectral: bool = True):
        super().__init__()

        self.key_dim = max(1, dim // 4) 
        
        self.q  = _sn_linear(dim, self.key_dim, spectral)
        self.k  = _sn_linear(dim, self.key_dim, spectral)
        self.v  = _sn_linear(dim, dim,          spectral)
        
        # Escala correta baseada na dimensão que entra no produto escalar: 1 / sqrt(key_dim)
        self.scale = self.key_dim ** -0.5
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, C)  →  (B, N, C)"""
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return x + attn @ v          # residual


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale PointNet Branch
# ─────────────────────────────────────────────────────────────────────────────

class _PointBranch(nn.Module):
    """One branch of the multi-scale feature extractor."""

    def __init__(self, in_ch: int, hidden: int, spectral: bool):
        super().__init__()
        self.net = nn.Sequential(
            _sn_linear(in_ch,  hidden,      spectral), nn.LeakyReLU(0.2, True),
            _sn_linear(hidden, hidden * 2,  spectral), nn.LeakyReLU(0.2, True),
            _sn_linear(hidden * 2, hidden * 4, spectral), nn.LeakyReLU(0.2, True),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts : (B, N, C)  →  (B, hidden*4)"""
        return self.net(pts).max(dim=1).values


# ─────────────────────────────────────────────────────────────────────────────
# Improved Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class PointNetDiscriminator(nn.Module):
    """
    Improved PointNet-style discriminator for point clouds shaped (B, N, 6).

    Key upgrades vs. the baseline:
    ──────────────────────────────
    • Mini T-Net aligns XYZ before feature extraction.
    • Three parallel multi-scale branches (narrow / mid / wide).
    • Lightweight self-attention before global max-pool.
    • Spectral norm throughout.
    • ``compute_gradient_penalty`` classmethod for WGAN-GP / R1.

    Args
    ────
    in_ch        : input channels per point  (6 = XYZ + normals)
    base_ch      : base channel width; branches use base_ch, base_ch*2,
                   base_ch*4 as their 'hidden' width  →  concat dim is
                   (base_ch + base_ch*2 + base_ch*4) * 4
    use_spectral : apply spectral norm on every trainable layer
    use_tnet     : prepend a 3-D spatial transformer (T-Net)
    """

    def __init__(
        self,
        in_ch: int        = 6,
        base_ch: int      = 64,
        use_spectral: bool = True,
        use_tnet: bool     = True,
    ):
        super().__init__()
        self.use_tnet = use_tnet

        # ── Optional T-Net ────────────────────────────────────────────────
        if use_tnet:
            self.tnet = _MiniTNet(spectral=use_spectral)

        # ── Multi-Scale Branches ──────────────────────────────────────────
        # Branch widths: narrow (base_ch), mid (base_ch*2), wide (base_ch*4)
        # Each branch outputs  width * 4  channels after pooling.
        self.branch_narrow = _PointBranch(in_ch, base_ch,       use_spectral)
        self.branch_mid    = _PointBranch(in_ch, base_ch * 2,   use_spectral)
        self.branch_wide   = _PointBranch(in_ch, base_ch * 4,   use_spectral)

        # Concatenated feature dim:
        # (base_ch + 2*base_ch + 4*base_ch) * 4  =  7 * base_ch * 4 = 28*base_ch
        fused_dim = (base_ch + base_ch * 2 + base_ch * 4) * 4   # 28 * base_ch

        # ── Self-Attention over fused_dim (treated as 1 token) ────────────
        # We expand to a small sequence of 4 virtual tokens for the attention
        # to have something meaningful to attend over.
        self.attn_proj = _sn_linear(fused_dim, fused_dim, use_spectral)
        # 1-D self-attention expects (B, S, C); we'll use S=4, C=fused_dim//4
        attn_tok_dim = fused_dim // 4
        self.attn = _SelfAttention1D(attn_tok_dim, spectral=use_spectral)
        self.attn_merge = _sn_linear(fused_dim, fused_dim, use_spectral)

        # ── Classification Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            _sn_linear(fused_dim,  fused_dim // 2, use_spectral),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            _sn_linear(fused_dim // 2, fused_dim // 8, use_spectral),
            nn.LeakyReLU(0.2, True),
            _sn_linear(fused_dim // 8, 1, use_spectral),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        self.last_tnet_mat: torch.Tensor | None = None

        # 1. Alinha XYZ com T-Net
        if self.use_tnet:
            xyz = pts[..., :3]
            aligned_xyz, tnet_mat = self.tnet(xyz)
            self.last_tnet_mat = tnet_mat
            pts = torch.cat([aligned_xyz, pts[..., 3:]], dim=-1)

        # 2. Extração multi-escala
        f_n = self.branch_narrow(pts)
        f_m = self.branch_mid(pts)
        f_wide = self.branch_wide(pts)

        # Concatena as features globais -> Formato: (B, fused_dim)
        fused = torch.cat([f_n, f_m, f_wide], dim=-1)
        
        # Projeta as features
        fused = self.attn_proj(fused)

        # 3. FIX DA ATENÇÃO: Transforma o vetor em uma sequência de 4 tokens virtuais: (B, 4, fused_dim // 4)
        B, C = fused.shape
        fused_seq = fused.view(B, 4, C // 4)
        
        # Aplica a atenção na sequência 3D
        attn_out = self.attn(fused_seq)
        
        # Desfaz o mapeamento de volta para o formato plano 2D: (B, fused_dim)
        fused = attn_out.view(B, C)
        fused = self.attn_merge(fused)

        # 4. Cabeça de classificação
        return self.head(fused)
    # ── Gradient Penalty ──────────────────────────────────────────────────────

    @staticmethod
    def compute_gradient_penalty(
        discriminator: "PointNetDiscriminator",
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """
        WGAN-GP gradient penalty.

        ``(||∇D(x̂)||₂ − 1)²``  where  x̂ = α·real + (1−α)·fake

        Usage in the training loop::

            gp = PointNetDiscriminator.compute_gradient_penalty(
                model.discriminator, real_pts, fake_pts, lambda_gp=10.0
            )
            loss_D = loss_D_bce + gp

        Args
        ────
        discriminator : the D module
        real          : (B, N, 6) real samples
        fake          : (B, N, 6) generated samples (detached)
        lambda_gp     : penalty coefficient (default 10.0 from WGAN-GP paper)

        Returns
        ───────
        Scalar gradient-penalty loss (already multiplied by lambda_gp).
        """
        B = real.size(0)
        device = real.device

        alpha = torch.rand(B, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

        d_interp = discriminator(interpolated)

        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]                                       # (B, N, 6)

        grads_norm = grads.reshape(B, -1).norm(2, dim=1)    # (B,)
        gp = lambda_gp * ((grads_norm - 1) ** 2).mean()
        return gp