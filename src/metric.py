"""
metrics.py
----------
Differentiable and evaluation metrics for point-cloud VAE.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Internal helpers
# ============================================================

def _pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """
    a : (B, M, 3)
    b : (B, N, 3)
    returns: (B, M, N)  squared Euclidean distances
    """
    a2 = (a ** 2).sum(dim=2, keepdim=True)
    b2 = (b ** 2).sum(dim=2, keepdim=True)
    ab = torch.bmm(a, b.transpose(1, 2))
    return (a2 + b2.transpose(1, 2) - 2 * ab).clamp(min=0.0)


# ============================================================
# Chamfer Distance
# ============================================================

def chamfer_distance(pred: Tensor, target: Tensor, reduce: str = "mean"):
    sq = _pairwise_sq_dist(pred, target)
    cd_pred = sq.min(dim=2).values
    cd_tgt  = sq.min(dim=1).values

    if reduce == "none":
        return cd_pred.mean(1) + cd_tgt.mean(1), cd_pred.mean(1), cd_tgt.mean(1)

    agg = torch.mean if reduce == "mean" else torch.sum
    return agg(cd_pred) + agg(cd_tgt), agg(cd_pred), agg(cd_tgt)


def chamfer_distance_knn(pred: Tensor, target: Tensor, k: int = 1, reduce: str = "mean"):
    with torch.autocast(device_type=pred.device.type, enabled=False):
        dist = torch.cdist(pred.float(), target.float())

    cd_pred = dist.topk(k, dim=2, largest=False).values.mean(dim=2)
    cd_tgt  = dist.topk(k, dim=1, largest=False).values.mean(dim=1)

    if reduce == "none":
        return cd_pred.mean(1) + cd_tgt.mean(1), cd_pred.mean(1), cd_tgt.mean(1)

    agg = torch.mean if reduce == "mean" else torch.sum
    return agg(cd_pred) + agg(cd_tgt), agg(cd_pred), agg(cd_tgt)


# ============================================================
# Approximate EMD (Sinkhorn)
# ============================================================

def emd_approx(pred: Tensor, target: Tensor, n_iters: int = 50, eps: float = 0.05, reduce: str = "mean") -> Tensor:
    B, N, _ = pred.shape
    assert pred.shape == target.shape

    cost = torch.cdist(pred.float(), target.float())

    log_a = torch.full((B, N), -torch.log(torch.tensor(float(N))), device=pred.device)
    log_b = log_a.clone()
    log_u = torch.zeros_like(log_a)
    log_K = -cost / eps

    for _ in range(n_iters):
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)

    log_T = log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    emd = (log_T.exp() * cost).sum(dim=(1, 2))

    if reduce == "none":
        return emd
    return emd.mean() if reduce == "mean" else emd.sum()


# ============================================================
# F-Score
# ============================================================

def f_score(pred: Tensor, target: Tensor, threshold: float = 0.01, reduce: str = "mean") -> dict[str, Tensor]:
    dist = torch.cdist(pred.float(), target.float())
    prec = (dist.min(dim=2).values < threshold).float().mean(dim=1)
    rec  = (dist.min(dim=1).values < threshold).float().mean(dim=1)
    fs   = 2 * prec * rec / (prec + rec).clamp(min=1e-8)

    if reduce == "none":
        return {"precision": prec, "recall": rec, "f_score": fs}
    return {"precision": prec.mean(), "recall": rec.mean(), "f_score": fs.mean()}


# ============================================================
# Normal Consistency
# ============================================================

def normal_consistency(pred_pts: Tensor, pred_nrm: Tensor, tgt_pts: Tensor, tgt_nrm: Tensor, reduce: str = "mean") -> Tensor:
    pred_nrm = F.normalize(pred_nrm, dim=2)
    tgt_nrm  = F.normalize(tgt_nrm,  dim=2)

    dist   = torch.cdist(pred_pts.float(), tgt_pts.float())
    nn_idx = dist.min(dim=2).indices
    nn_nrm = tgt_nrm.gather(1, nn_idx.unsqueeze(2).expand(-1, -1, 3))

    cos_sim = (pred_nrm * nn_nrm).sum(dim=2).abs()

    if reduce == "none":
        return cos_sim.mean(dim=1)
    return cos_sim.mean() if reduce == "mean" else cos_sim.sum()


# ============================================================
# KL Divergence
# ============================================================

def kl_divergence(mu: Tensor, logvar: Tensor, free_bits: float = 0.0, reduce: str = "mean") -> Tensor:
    """
    KL[q(z|x) || p(z)] — matches LION's vae_adain.py get_loss().

    LION does NOT use free-bits: KL is computed raw per element and then
    averaged over all latent dims (N and D for the local cloud, D for style).
    free_bits=0.0 by default to match this; pass free_bits>0 only if you
    explicitly want the original behaviour.
    """
    logvar = torch.clamp(logvar, -10.0, 10.0)
    mu     = torch.clamp(mu,     -10.0, 10.0)

    # Element-wise KL: (B, ...) same shape as input
    kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, [N,] D)

    if free_bits > 0.0:
        kl_elem = torch.clamp(kl_elem, min=free_bits)

    # Reduce over all latent dims — mean keeps scale invariant to D and N
    if kl_elem.dim() == 3:          # (B, N, D)  — local latent cloud
        kl_per_sample = kl_elem.mean(dim=(1, 2))   # (B,)
    else:                           # (B, D)     — global style vector
        kl_per_sample = kl_elem.mean(dim=1)        # (B,)

    if reduce == "none":
        return kl_per_sample
    return kl_per_sample.mean() if reduce == "mean" else kl_per_sample.sum()


# ============================================================
# Full VAE ELBO loss
# ============================================================

def vae_loss(
    pred_xyz:      Tensor,
    target_xyz:    Tensor,
    mu_points:     Tensor,
    logvar_points: Tensor,
    mu_style:      Tensor,
    logvar_style:  Tensor,
    beta:          float          = 1.0,
    # LION-style separate KL weights (mirrors weight_kl_pt / weight_kl_feat / weight_kl_glb):
    # position channels of z_l (first 3) — typically 0 or very small; the anchor provides
    # the main training signal, so KL here fights reconstruction.
    beta_pos_pts:  float          = 0.0,
    # feature channels of z_l (remaining latent_dim) — mild regularisation helps the
    # diffusion model; can be set to 0.001–0.01 without hurting reconstruction.
    beta_feat_pts: float          = 0.001,
    # z_g KL — keep at 1.0; the first-stage DDPM generates z_g from N(0,I) directly.
    beta_style:    float          = 1.0,
    recon_loss:    str            = "chamfer",
    emd_weight:    float          = 0.5,
    emd_iters:     int            = 30,
    normals_pred:  Tensor | None  = None,
    normal_target: Tensor | None  = None,
    normal_weight: float          = 0.0,
) -> dict[str, Tensor]:

    # --- 1. Reconstruction loss ----------------------------------------
    if recon_loss == "chamfer":
        recon, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        out = {"recon": recon, "cd_forward": cd_f, "cd_backward": cd_b}

    elif recon_loss == "emd":
        # FIX #3: EMD requires equal point counts — fall back if sizes differ
        if pred_xyz.shape[1] != target_xyz.shape[1]:
            # subsample / upsample target to match pred
            B, N_pred, _ = pred_xyz.shape
            idx = torch.randperm(target_xyz.shape[1], device=target_xyz.device)[:N_pred]
            target_emd = target_xyz[:, idx, :]
        else:
            target_emd = target_xyz
        recon = emd_approx(pred_xyz, target_emd, n_iters=emd_iters)
        out   = {"recon": recon}

    elif recon_loss == "both":
        cd, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        # Sinkhorn cost matrix is O(N²) — subsample to 512 pts to stay in memory.
        # 512 points is sufficient for a reliable EMD gradient signal.
        N_emd = min(pred_xyz.shape[1], 512)
        idx_p = torch.randperm(pred_xyz.shape[1],  device=pred_xyz.device)[:N_emd]
        idx_t = torch.randperm(target_xyz.shape[1], device=target_xyz.device)[:N_emd]
        emd_raw = emd_approx(pred_xyz[:, idx_p], target_xyz[:, idx_t], n_iters=emd_iters)
        # Sinkhorn uses uniform marginals 1/N, so emd_raw ≈ mean transport
        # distance per point — same scale as Chamfer. No division needed.
        emd = emd_raw
        recon = (1 - emd_weight) * cd + emd_weight * emd
        out   = {"recon": recon, "cd": cd, "emd": emd,
                 "cd_forward": cd_f, "cd_backward": cd_b}
    else:
        raise ValueError(f"Unknown recon_loss: '{recon_loss}'.")

    # --- 2. Normal loss (optional) -------------------------------------
    if normal_weight > 0.0 and normals_pred is not None and normal_target is not None:
        n_consistency = normal_consistency(pred_xyz, normals_pred, target_xyz, normal_target)
        n_loss = (1.0 - n_consistency).clamp(min=0.0, max=1.0)
        out["normal_loss"] = n_loss
    else:
        n_loss = pred_xyz.new_zeros(1)
        out["normal_loss"] = n_loss

    # --- 3. KL losses --------------------------------------------------
    # mu_points is (B, N, 3 + latent_dim): split along the last dim into
    #   position channels [:, :, :3]      — anchored to input_xyz by the encoder
    #   feature  channels [:, :, 3:]      — abstract latents toward N(0,I)
    # This mirrors LION's weight_kl_pt / weight_kl_feat separation.
    kl_pos  = kl_divergence(mu_points[:, :, :3],  logvar_points[:, :, :3])
    if mu_points.shape[-1] > 3:
        kl_feat = kl_divergence(mu_points[:, :, 3:], logvar_points[:, :, 3:])
    else:
        kl_feat = mu_points.new_zeros(1)
    kl_sty  = kl_divergence(mu_style, logvar_style)

    out["kl_pos"]    = kl_pos
    out["kl_feat"]   = kl_feat
    out["kl_points"] = kl_pos + kl_feat   # total local KL for monitoring
    out["kl_style"]  = kl_sty

    # --- 4. Total ELBO -------------------------------------------------
    total = (
        recon
        + normal_weight * n_loss
        + beta * beta_pos_pts  * kl_pos
        + beta * beta_feat_pts * kl_feat
        + beta * beta_style    * kl_sty
    )
    out["total"] = total
    return out


# ============================================================
# Generation quality metrics  (MMD, COV, 1-NNA)
# ============================================================
# These follow the standard evaluation protocol from:
#   "Learning Representations and Generative Models for 3D Point Clouds"
#   (Achlioptas et al., ICML 2018)
#
# All three use pairwise Chamfer distances between two sets of clouds:
#   gen   — generated clouds  (M, N, 3)
#   ref   — reference/real   (R, N, 3)
#
# They measure the triangle of desiderata: quality, diversity, fidelity.

def _pairwise_cd_matrix(a: Tensor, b: Tensor, batch_size: int = 64) -> Tensor:
    """
    Compute pairwise Chamfer distances between every cloud in a and every cloud in b.
    Returns (len(a), len(b)) matrix.  Batched to avoid OOM on large sets.
    """
    M, R = len(a), len(b)
    dist = torch.zeros(M, R, device=a.device)
    for i in range(0, M, batch_size):
        ai = a[i:i + batch_size]               # (bs, N, 3)
        for j in range(0, R, batch_size):
            bj = b[j:j + batch_size]           # (bs2, N, 3)
            # broadcast: compare each cloud in ai against each in bj
            cd_vals = []
            for k in range(len(ai)):
                ak = ai[k].unsqueeze(0).expand(len(bj), -1, -1)   # (bs2, N, 3)
                cd, _, _ = chamfer_distance_knn(ak, bj, reduce="none")
                cd_vals.append(cd)             # (bs2,)
            dist[i:i + batch_size, j:j + batch_size] = torch.stack(cd_vals)
    return dist


@torch.no_grad()
def generation_metrics(
    gen: Tensor,
    ref: Tensor,
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Compute MMD, COV, and 1-NNA between generated and reference point clouds.

    Parameters
    ----------
    gen : (M, N, 3)  generated clouds
    ref : (R, N, 3)  reference (real) clouds — ideally a held-out test split
    batch_size : chunk size for pairwise CD computation

    Returns
    -------
    dict with keys:
        mmd   — Minimum Matching Distance (lower = better quality)
        cov   — Coverage (higher = better diversity, range [0,1])
        nna   — 1-Nearest-Neighbour Accuracy (closer to 0.5 = better)
    """
    gen = gen.float();  ref = ref.float()

    # Pairwise CD matrices
    cd_gen_ref = _pairwise_cd_matrix(gen, ref, batch_size)   # (M, R)
    cd_ref_gen = cd_gen_ref.T                                  # (R, M)

    # ---- MMD: for each ref cloud, find closest generated cloud ----------
    mmd = cd_ref_gen.min(dim=1).values.mean().item()

    # ---- COV: fraction of ref clouds matched by at least one gen cloud --
    matched = cd_ref_gen.min(dim=1).indices                   # (R,) — closest gen per ref
    cov = matched.unique().numel() / len(ref)

    # ---- 1-NNA ----------------------------------------------------------
    # Build the full (M+R) x (M+R) pairwise CD matrix in blocks.
    # Label: gen=0, ref=1.  For each cloud, its 1-NN (excluding itself)
    # should have the same label if gen and ref are indistinguishable.
    cd_gen_gen = _pairwise_cd_matrix(gen, gen, batch_size)    # (M, M)
    cd_ref_ref = _pairwise_cd_matrix(ref, ref, batch_size)    # (R, R)

    # For gen samples: find NN in gen∪ref (excluding self)
    cd_gen_gen.fill_diagonal_(float("inf"))
    nn_gen_in_gen = cd_gen_gen.min(dim=1).values              # (M,)
    nn_gen_in_ref = cd_gen_ref.min(dim=1).values              # (M,)
    # NN of each gen cloud is in gen if nn_gen < nn_ref
    gen_correct = (nn_gen_in_gen < nn_gen_in_ref).float()

    # For ref samples: find NN in gen∪ref (excluding self)
    cd_ref_ref.fill_diagonal_(float("inf"))
    nn_ref_in_ref = cd_ref_ref.min(dim=1).values              # (R,)
    nn_ref_in_gen = cd_ref_gen.min(dim=1).values              # (R,)
    ref_correct = (nn_ref_in_ref < nn_ref_in_gen).float()

    nna = torch.cat([gen_correct, ref_correct]).mean().item()

    return {"mmd": mmd, "cov": cov, "nna": nna}