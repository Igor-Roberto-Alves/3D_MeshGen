"""
metrics.py
----------
Differentiable and evaluation metrics for point-cloud VAE.

Metrics
-------
chamfer_distance     – bidirectional Chamfer Distance (differentiable, O(N²))
chamfer_distance_knn – fast Chamfer via kNN (differentiable for large N)
emd_approx           – Sinkhorn-based approximate Earth Mover's Distance
f_score              – precision / recall / F-score at threshold τ
normal_consistency   – mean cosine similarity between nearest-neighbour normals
kl_divergence        – closed-form KL for Gaussian VAE prior
vae_loss             – full ELBO = recon_loss + β·KL
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Internal helpers
# ============================================================

def _pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute squared pairwise Euclidean distances.

    a : (B, M, 3)
    b : (B, N, 3)
    returns: (B, M, N)
    """
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    a2 = (a ** 2).sum(dim=2, keepdim=True)   # (B, M, 1)
    b2 = (b ** 2).sum(dim=2, keepdim=True)   # (B, N, 1)
    ab = torch.bmm(a, b.transpose(1, 2))     # (B, M, N)
    return (a2 + b2.transpose(1, 2) - 2 * ab).clamp(min=0.0)


# ============================================================
# Chamfer Distance
# ============================================================

def chamfer_distance(
    pred: Tensor,
    target: Tensor,
    reduce: str = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Bidirectional Chamfer Distance (O(BNM) – suitable for N ≤ 2048).

    Parameters
    ----------
    pred, target : (B, N, 3)
    reduce       : "mean" | "sum" | "none"

    Returns
    -------
    cd_total : scalar (or (B,) if reduce="none")
    cd_pred  : pred → target (forward)
    cd_tgt   : target → pred (backward)
    """
    sq = _pairwise_sq_dist(pred, target)              # (B, N, N)

    # each pred point → nearest target
    cd_pred = sq.min(dim=2).values                    # (B, N)
    # each target point → nearest pred
    cd_tgt  = sq.min(dim=1).values                    # (B, N)

    if reduce == "none":
        return cd_pred.mean(1) + cd_tgt.mean(1), cd_pred.mean(1), cd_tgt.mean(1)

    agg = torch.mean if reduce == "mean" else torch.sum
    cd_pred_s = agg(cd_pred)
    cd_tgt_s  = agg(cd_tgt)
    return cd_pred_s + cd_tgt_s, cd_pred_s, cd_tgt_s


# ============================================================
# Fast Chamfer via kNN  (memory-friendly for large point clouds)
# ============================================================

def chamfer_distance_knn(
    pred: Tensor,
    target: Tensor,
    k: int = 1,
    reduce: str = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Chamfer Distance using torch.cdist (uses less peak memory).

    Parameters are identical to chamfer_distance.
    """
    dist = torch.cdist(pred.float(), target.float())                  # (B, N, M)

    cd_pred = dist.topk(k, dim=2, largest=False).values.mean(dim=2)  # (B, N)
    cd_tgt  = dist.topk(k, dim=1, largest=False).values.mean(dim=1)  # (B, M)

    if reduce == "none":
        return cd_pred.mean(1) + cd_tgt.mean(1), cd_pred.mean(1), cd_tgt.mean(1)

    agg = torch.mean if reduce == "mean" else torch.sum
    cd_pred_s = agg(cd_pred)
    cd_tgt_s  = agg(cd_tgt)
    return cd_pred_s + cd_tgt_s, cd_pred_s, cd_tgt_s


# ============================================================
# Approximate Earth Mover's Distance  (Sinkhorn)
# ============================================================

def emd_approx(
    pred: Tensor,
    target: Tensor,
    n_iters: int = 50,
    eps: float = 0.05,
    reduce: str = "mean",
) -> Tensor:
    """
    Approximate EMD via Sinkhorn iterations (differentiable).

    pred, target : (B, N, 3)   – must have the same N
    n_iters      : Sinkhorn iterations
    eps          : entropy regularisation

    Returns scalar loss.
    """
    B, N, _ = pred.shape
    assert pred.shape == target.shape, "pred and target must have the same shape."

    cost = torch.cdist(pred.float(), target.float())                  # (B, N, N)

    # uniform marginals
    log_a = torch.full((B, N), -torch.log(torch.tensor(float(N))),
                       device=pred.device)
    log_b = log_a.clone()

    # log-domain Sinkhorn
    log_u = torch.zeros_like(log_a)
    log_K = -cost / eps

    for _ in range(n_iters):
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)

    # transport plan
    log_T = log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1)   # (B, N, N)
    T = log_T.exp()

    emd = (T * cost).sum(dim=(1, 2))                          # (B,)

    if reduce == "none":
        return emd
    return emd.mean() if reduce == "mean" else emd.sum()


# ============================================================
# F-Score
# ============================================================

def f_score(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.01,
    reduce: str = "mean",
) -> dict[str, Tensor]:
    """
    F-Score at distance threshold τ.

    Parameters
    ----------
    pred, target : (B, N, 3)
    threshold    : τ in the same unit as the point coordinates
    reduce       : "mean" | "none"

    Returns
    -------
    dict with keys "precision", "recall", "f_score"
    """
    dist = torch.cdist(pred.float(), target.float())                   # (B, N, M)

    # precision: fraction of pred points with a match within τ
    prec = (dist.min(dim=2).values < threshold).float().mean(dim=1)   # (B,)
    # recall: fraction of target points covered
    rec  = (dist.min(dim=1).values < threshold).float().mean(dim=1)   # (B,)

    denom = (prec + rec).clamp(min=1e-8)
    fs    = 2 * prec * rec / denom                                     # (B,)

    if reduce == "none":
        return {"precision": prec, "recall": rec, "f_score": fs}

    return {
        "precision": prec.mean(),
        "recall":    rec.mean(),
        "f_score":   fs.mean(),
    }


# ============================================================
# Normal Consistency
# ============================================================

def normal_consistency(
    pred_pts:  Tensor,
    pred_nrm:  Tensor,
    tgt_pts:   Tensor,
    tgt_nrm:   Tensor,
    reduce:    str = "mean",
) -> Tensor:
    """
    Mean absolute cosine similarity between each predicted point's normal and
    its nearest neighbour's normal in the target cloud.

    pred_pts, tgt_pts : (B, N, 3)
    pred_nrm, tgt_nrm : (B, N, 3)  – not necessarily unit-length
    """
    pred_nrm = F.normalize(pred_nrm, dim=2)
    tgt_nrm  = F.normalize(tgt_nrm,  dim=2)

    dist = torch.cdist(pred_pts.float(), tgt_pts.float())             # (B, N, M)
    nn_idx = dist.min(dim=2).indices                  # (B, N)

    # gather nearest target normals
    nn_idx_exp  = nn_idx.unsqueeze(2).expand(-1, -1, 3)
    nn_nrm      = tgt_nrm.gather(1, nn_idx_exp)      # (B, N, 3)

    cos_sim = (pred_nrm * nn_nrm).sum(dim=2).abs()   # (B, N)

    if reduce == "none":
        return cos_sim.mean(dim=1)
    return cos_sim.mean() if reduce == "mean" else cos_sim.sum()


# ============================================================
# KL Divergence  (Gaussian VAE)
# ============================================================

def kl_divergence(mu: Tensor, logvar: Tensor, reduce: str = "mean") -> Tensor:
    """
    Closed-form KL[q(z|x) || p(z)]  where p(z) = N(0, I).

    KL = -0.5 * Σ (1 + logvar - μ² - exp(logvar))

    mu, logvar : (B, latent_dim)
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, latent_dim)
    kl = kl.sum(dim=1)                                     # (B,)

    if reduce == "none":
        return kl
    return kl.mean() if reduce == "mean" else kl.sum()


# ============================================================
# Full VAE ELBO loss
# ============================================================

def vae_loss(
    pred_xyz:   Tensor,
    target_xyz: Tensor,
    mu:         Tensor,
    logvar:     Tensor,
    beta:       float  = 1.0,
    recon_loss: str    = "chamfer",
    # optional – used only when recon_loss == "both"
    emd_weight: float  = 0.5,
    emd_iters:  int    = 30,
) -> dict[str, Tensor]:
    """
    ELBO = reconstruction_loss + β · KL

    Parameters
    ----------
    pred_xyz    : (B, N, 3)  – reconstructed coordinates
    target_xyz  : (B, N, 3)  – ground-truth coordinates
    mu, logvar  : (B, latent_dim)
    beta        : KL weight  (β-VAE)
    recon_loss  : "chamfer" | "emd" | "both"
    emd_weight  : weight of EMD when recon_loss == "both"

    Returns
    -------
    dict with keys "total", "recon", "kl"  (and optionally "cd", "emd")
    """
    # --- reconstruction -----------------------------------------------
    if recon_loss == "chamfer":
        recon, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        out = {"recon": recon, "cd_forward": cd_f, "cd_backward": cd_b}

    elif recon_loss == "emd":
        recon = emd_approx(pred_xyz, target_xyz, n_iters=emd_iters)
        out   = {"recon": recon}

    elif recon_loss == "both":
        cd, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        emd            = emd_approx(pred_xyz, target_xyz, n_iters=emd_iters)
        recon = (1 - emd_weight) * cd + emd_weight * emd
        out   = {"recon": recon, "cd": cd, "emd": emd,
                 "cd_forward": cd_f, "cd_backward": cd_b}
    else:
        raise ValueError(f"Unknown recon_loss: '{recon_loss}'. "
                         f"Choose from 'chamfer', 'emd', 'both'.")

    # --- KL ------------------------------------------------------------
    kl = kl_divergence(mu, logvar)

    total = recon + beta * kl
    out.update({"total": total, "kl": kl})
    return out