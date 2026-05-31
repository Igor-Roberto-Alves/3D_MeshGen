def chamfer_distance(pred, target):
    # pred, target: (B, N, 6)
    # Use only XYZ for neighbor lookup, but compute loss on all 6 dims
    pred_xyz   = pred[..., :3]
    target_xyz = target[..., :3]

    diff = pred_xyz.unsqueeze(2) - target_xyz.unsqueeze(1)   # (B, N, N, 3)
    dist = diff.pow(2).sum(-1)                                # (B, N, N)

    # Nearest neighbor indices from XYZ
    nn_pred_to_tgt = dist.argmin(dim=2)   # (B, N)
    nn_tgt_to_pred = dist.argmin(dim=1)   # (B, N)

    # Gather full 6D neighbors
    idx1 = nn_pred_to_tgt.unsqueeze(-1).expand(-1, -1, 6)
    idx2 = nn_tgt_to_pred.unsqueeze(-1).expand(-1, -1, 6)

    matched_tgt  = target.gather(1, idx1)   # (B, N, 6)
    matched_pred = pred.gather(1, idx2)     # (B, N, 6)

    cd = (pred - matched_tgt).pow(2).sum(-1).mean() + \
         (target - matched_pred).pow(2).sum(-1).mean()
    return cd