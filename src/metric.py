import torch

def chamfer_distance(pred, target, normal_weight=0.5):
    """
    Versão Otimizada e Corrigida da Perda de Chamfer (XYZ + Normais).
    Consumo de memória reduzido via expansão algébrica e livre de bugs de alinhamento.
    
    pred, target: (B, N, 6) -> [X, Y, Z, Nx, Ny, Nz]
    """
    B, N, _ = pred.shape
    device = pred.device

    pred_xyz   = pred[..., :3]
    target_xyz = target[..., :3]
    pred_norm  = pred[..., 3:]
    target_norm = target[..., 3:]

    # 1. Distância Euclidiana Otimizada (Evita OOM na GPU)
    r_pred = torch.sum(pred_xyz ** 2, dim=-1, keepdim=True) 
    r_tgt  = torch.sum(target_xyz ** 2, dim=-1, keepdim=True)
    mul    = torch.bmm(pred_xyz, target_xyz.transpose(1, 2))
    dist   = r_pred - 2 * mul + r_tgt.transpose(1, 2)
    
    # Garantir que imprecisões numéricas não criem distâncias negativas antes do argmin
    dist = torch.clamp(dist, min=0.0)

    # 2. Encontrar os vizinhos mais próximos (Índices)
    nn_pred_to_tgt = dist.argmin(dim=2)   # (B, N)
    nn_tgt_to_pred = dist.argmin(dim=1)   # (B, N)

    # 3. Coleta dos vizinhos geométricos correspondentes (XYZ)
    idx_xyz = nn_pred_to_tgt.unsqueeze(-1).expand(-1, -1, 3)
    idx_pred = nn_tgt_to_pred.unsqueeze(-1).expand(-1, -1, 3)
    
    matched_tgt_xyz  = target_xyz.gather(1, idx_xyz)
    matched_pred_xyz = pred_xyz.gather(1, idx_pred)

    # Perda Geométrica (XYZ)
    cd_xyz = (pred_xyz - matched_tgt_xyz).pow(2).sum(-1).mean() + \
             (target_xyz - matched_pred_xyz).pow(2).sum(-1).mean()

    # 4. Coleta e Alinhamento das Normais correspondentes
    matched_tgt_norm  = target_norm.gather(1, idx_xyz)
    matched_pred_norm = pred_norm.gather(1, idx_pred)

    # Normalização dos vetores para garantir Cosseno Perfeito
    pred_norm_u = pred_norm / (pred_norm.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm_u = target_norm / (target_norm.norm(dim=-1, keepdim=True) + 1e-8)
    
    matched_tgt_norm_u  = matched_tgt_norm / (matched_tgt_norm.norm(dim=-1, keepdim=True) + 1e-8)
    matched_pred_norm_u = matched_pred_norm / (matched_pred_norm.norm(dim=-1, keepdim=True) + 1e-8)

    # Cálculo da perda de orientação (1.0 - Cosine_Similarity)
    normal_loss_pred = 1.0 - (pred_norm_u * matched_tgt_norm_u).sum(-1)
    normal_loss_tgt  = 1.0 - (matched_pred_norm_u * target_norm_u).sum(-1)

    cd_normal = normal_loss_pred.mean() + normal_loss_tgt.mean()

    # Combinar ambas com o peso estipulado
    return cd_xyz + (normal_weight * cd_normal)