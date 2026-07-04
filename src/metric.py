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

def emd_approx(pred: Tensor, target: Tensor, n_iters: int = 30, eps: float = 0.05,
               reduce: str = "mean", n_subsample: int = 0) -> Tensor:
    B, N, _ = pred.shape
    assert pred.shape == target.shape

    # Subsample for speed: O(N^2) cost matrix → O(n_subsample^2)
    if 0 < n_subsample < N:
        idx    = torch.randperm(N, device=pred.device)[:n_subsample]
        pred   = pred[:, idx]
        target = target[:, idx]
        N      = n_subsample

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
# Generative-model set metrics (MMD-CD / COV-CD / 1-NNA-CD)
# ============================================================
#
# Unlike the losses above (which score a single prediction against its own
# target), these score a *set* of generated point clouds against a held-out
# *set* of real ones — i.e. whether the generator's output distribution
# matches the real one, not just whether individual samples denoise well.
# A model can have a great denoising loss and still mode-collapse or produce
# off-distribution shapes; these are the standard metrics (PointFlow, used by
# LION) for catching that when picking a checkpoint.

def _pairwise_cd_cross(a: Tensor, b: Tensor, chunk: int = 4) -> Tensor:
    """
    a: (Ma, N, 3)   b: (Mb, N, 3)   →   (Ma, Mb) Chamfer-distance matrix.

    Computed in row chunks (`chunk` clouds of `a` against all of `b` per
    step) instead of building the full Ma*Mb x N x N tensor at once, to
    bound GPU memory regardless of how many samples are requested.
    """
    Ma, Mb = a.shape[0], b.shape[0]
    D = a.new_zeros(Ma, Mb)
    for start in range(0, Ma, chunk):
        end  = min(start + chunk, Ma)
        rows = a[start:end]
        c    = rows.shape[0]
        rows_exp = rows.unsqueeze(1).expand(c, Mb, -1, -1).reshape(c * Mb, *a.shape[1:])
        cols_exp = b.unsqueeze(0).expand(c, Mb, -1, -1).reshape(c * Mb, *b.shape[1:])
        cd, _, _ = chamfer_distance_knn(rows_exp, cols_exp, reduce="none")
        D[start:end] = cd.view(c, Mb)
    return D


def mmd_cov(gen: Tensor, ref: Tensor, chunk: int = 4) -> dict[str, Tensor]:
    """
    MMD-CD and COV-CD (Yang et al., PointFlow 2019 — Sec 5.1; also used by LION).

    gen: (Ng, N, 3)  generated point clouds
    ref: (Nr, N, 3)  held-out real point clouds, SAME normalised coordinate
                      space as gen (see Vae.normalize_pc — apply it to real
                      clouds before calling this, since the decoder's output
                      already lives in that space).

    MMD — for each real sample, distance to its closest generated sample,
          averaged over the real set. Lower = generated samples are on
          average closer to real ones (fidelity).
    COV  — fraction of real samples that are *someone's* nearest neighbour
          among the generated set. Lower = many real modes are never
          produced by any generated sample (mode collapse / low diversity).
    """
    D = _pairwise_cd_cross(gen, ref, chunk=chunk)          # D[i, j] = CD(gen_i, ref_j)
    mmd = D.min(dim=0).values.mean()                        # per ref: closest gen, avg over ref
    cov = D.min(dim=1).indices.unique().numel() / ref.shape[0]  # frac of ref hit as NN by some gen
    return {"mmd": mmd, "cov": gen.new_tensor(cov)}


def nna_1nn(gen: Tensor, ref: Tensor, chunk: int = 4) -> Tensor:
    """
    1-NNA-CD (Lopez-Paz & Oquab 2017, adapted by PointFlow for point clouds).

    Pools gen and ref into one set; for every sample, finds its nearest
    neighbour (leave-one-out) in the pooled set and checks whether that
    neighbour comes from the same set (gen or ref). Returns the resulting
    classification accuracy.

    0.5 == ideal: gen and ref are statistically indistinguishable by 1-NN.
    1.0 == gen and ref are trivially separable (bad — either the generator
           is off-distribution, or it's producing near-duplicates that
           cluster tightly instead of matching the real spread).
    """
    Ng, Nr = gen.shape[0], ref.shape[0]
    all_pts = torch.cat([gen, ref], dim=0)
    D = _pairwise_cd_cross(all_pts, all_pts, chunk=chunk)
    D.fill_diagonal_(float("inf"))                          # exclude self as its own neighbour

    labels  = torch.cat([gen.new_zeros(Ng), gen.new_ones(Nr)])
    nn_idx  = D.argmin(dim=1)
    correct = (labels[nn_idx] == labels).float().mean()
    return correct


def generative_metrics(gen: Tensor, ref: Tensor, chunk: int = 4) -> dict[str, Tensor]:
    """MMD-CD, COV-CD and 1-NNA-CD in one call. See mmd_cov / nna_1nn for definitions."""
    out = mmd_cov(gen, ref, chunk=chunk)
    out["nna_1nn"] = nna_1nn(gen, ref, chunk=chunk)
    return out


# ============================================================
# KL Divergence
# ============================================================

def kl_divergence(mu: Tensor, logvar: Tensor, free_bits: float = 0.5, reduce: str = "mean") -> Tensor:
    """
    KL[q(z|x) || p(z)] with free-bits per dimension to prevent posterior collapse.

    FIX #1: the original code used .sum(dim=(1,2)) / shape[1] which
    heavily under-weights the KL for the point cloud (B, N, D) case —
    dividing by N instead of N*D makes the KL ~D times too small,
    meaning beta effectively acts as beta/D.  We now use .mean() over
    all latent dimensions so the KL is dimensionally consistent with
    the reconstruction loss regardless of whether z is (B,D) or (B,N,D).

    FIX #2: free_bits threshold prevents posterior collapse.  Any
    dimension with KL < free_bits is treated as zero — the encoder is
    not penalised for using those dimensions.  Without this, the KL
    term can dominate early in training and force the encoder to ignore
    the input (mean-field collapse), which manifests as the decoder
    having to reconstruct from pure noise → cuboid outputs.
    """
    logvar = torch.clamp(logvar, -10.0, 10.0)
    mu     = torch.clamp(mu,     -10.0, 10.0)

    kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, [N,] D)

    if free_bits > 0.0:
        # Per-dimension free bits (Kingma et al. 2016):
        # average KL across the batch (and, for point latents, across points
        # too) for each latent dimension, then apply the floor. This prevents
        # collapse without locking individual samples.
        if kl_elem.dim() == 3:       # (B, N, D)
            kl_per_dim = kl_elem.mean(dim=(0, 1))      # (D,) — one floor per channel
            kl_per_dim = kl_per_dim.clamp(min=free_bits)
            kl_per_sample = kl_per_dim.mean().unsqueeze(0).expand(kl_elem.shape[0])
        else:                         # (B, D)
            kl_per_dim = kl_elem.mean(dim=0)           # (D,)
            kl_per_dim = kl_per_dim.clamp(min=free_bits)
            kl_per_sample = kl_per_dim.mean().unsqueeze(0).expand(kl_elem.shape[0])
    else:
        if kl_elem.dim() == 3:
            kl_per_sample = kl_elem.mean(dim=(1, 2))
        else:
            kl_per_sample = kl_elem.mean(dim=1)

    if reduce == "none":
        return kl_per_sample
    return kl_per_sample.mean() if reduce == "mean" else kl_per_sample.sum()


# ============================================================
# Full VAE ELBO loss
# ============================================================

def vae_loss(
    pred_xyz:        Tensor,
    target_xyz:      Tensor,
    mu_points:       Tensor,
    logvar_points:   Tensor,
    mu_style:        Tensor,
    logvar_style:    Tensor,
    beta:            float          = 1.0,
    beta_points:     float          = 1.0,
    beta_style:      float          = 1.0,
    free_bits:       float          = 0.0,
    recon_loss:      str            = "chamfer",
    emd_weight:      float          = 0.5,
    emd_iters:       int            = 15,
    emd_n_subsample: int            = 0,
    normals_pred:    Tensor | None  = None,
    normal_target:   Tensor | None  = None,
    normal_weight:   float          = 0.0,
) -> dict[str, Tensor]:

    # --- 1. Reconstruction loss ----------------------------------------
    if recon_loss == "chamfer":
        recon, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        out = {"recon": recon, "cd_forward": cd_f, "cd_backward": cd_b}

    elif recon_loss == "emd":
        if pred_xyz.shape[1] != target_xyz.shape[1]:
            B, N_pred, _ = pred_xyz.shape
            idx = torch.randperm(target_xyz.shape[1], device=target_xyz.device)[:N_pred]
            target_emd = target_xyz[:, idx, :]
        else:
            target_emd = target_xyz
        recon = emd_approx(pred_xyz, target_emd, n_iters=emd_iters, n_subsample=emd_n_subsample)
        out   = {"recon": recon}

    elif recon_loss == "both":
        cd, cd_f, cd_b = chamfer_distance_knn(pred_xyz, target_xyz)
        if pred_xyz.shape[1] != target_xyz.shape[1]:
            idx = torch.randperm(target_xyz.shape[1], device=target_xyz.device)[:pred_xyz.shape[1]]
            target_emd = target_xyz[:, idx, :]
        else:
            target_emd = target_xyz
        emd   = emd_approx(pred_xyz, target_emd, n_iters=emd_iters, n_subsample=emd_n_subsample)
        recon = (1 - emd_weight) * cd + emd_weight * emd
        out   = {"recon": recon, "cd": cd, "emd": emd, "cd_forward": cd_f, "cd_backward": cd_b}
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
    # mu_points / logvar_points are (B, N, latent_dim) — full local latent,
    # no xyz prefix, so we use the full tensor here.
    kl_pts = kl_divergence(mu_points, logvar_points, free_bits=free_bits)
    kl_sty = kl_divergence(mu_style,  logvar_style,  free_bits=free_bits)
    out["kl_points"] = kl_pts
    out["kl_style"]  = kl_sty

    # --- 4. Total ELBO -------------------------------------------------
    total = (
        recon
        + normal_weight * n_loss
        + beta * beta_points * kl_pts
        + beta * beta_style  * kl_sty
    )
    out["total"] = total
    return out