import torch


def chamfer_distance(pred, target, normal_weight=0.5):
    """
    Versão Estabilizada e Corrigida da Perda de Chamfer (XYZ + Normais).
    Aplica raiz quadrada antes da média para estabilização de gradientes (L2 linear).

    pred, target: (B, N, 6) -> [X, Y, Z, Nx, Ny, Nz]
    """
    B, N, _ = pred.shape
    device = pred.device

    pred_xyz = pred[..., :3]
    target_xyz = target[..., :3]
    pred_norm = pred[..., 3:]
    target_norm = target[..., 3:]

    # 1. Distância Euclidiana Otimizada (Matriz de Distâncias Cruzadas)
    r_pred = torch.sum(pred_xyz**2, dim=-1, keepdim=True)
    r_tgt = torch.sum(target_xyz**2, dim=-1, keepdim=True)
    mul = torch.bmm(pred_xyz, target_xyz.transpose(1, 2))
    dist = r_pred - 2 * mul + r_tgt.transpose(1, 2)

    # Garante estabilidade numérica contra valores ligeiramente negativos
    dist = torch.clamp(dist, min=0.0)

    # 2. Encontrar os índices dos vizinhos mais próximos
    nn_pred_to_tgt = dist.argmin(dim=2)  # (B, N)
    nn_tgt_to_pred = dist.argmin(dim=1)  # (B, N)

    # 3. Coleta dos vizinhos geométricos correspondentes (XYZ)
    idx_xyz = nn_pred_to_tgt.unsqueeze(-1).expand(-1, -1, 3)
    idx_pred = nn_tgt_to_pred.unsqueeze(-1).expand(-1, -1, 3)

    matched_tgt_xyz = target_xyz.gather(1, idx_xyz)
    matched_pred_xyz = pred_xyz.gather(1, idx_pred)

    # Perda Geométrica (XYZ)
    cd_xyz = (pred_xyz - matched_tgt_xyz).pow(2).sum(-1).mean() + (
        target_xyz - matched_pred_xyz
    ).pow(2).sum(-1).mean()

    # 4. Coleta e Alinhamento das Normais correspondentes
    matched_tgt_norm = target_norm.gather(1, idx_xyz)
    matched_pred_norm = pred_norm.gather(1, idx_pred)

    # Normalização dos vetores para garantir Cosseno Perfeito
    pred_norm_u = pred_norm / (pred_norm.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm_u = target_norm / (target_norm.norm(dim=-1, keepdim=True) + 1e-8)

    matched_tgt_norm_u = matched_tgt_norm / (
        matched_tgt_norm.norm(dim=-1, keepdim=True) + 1e-8
    )
    matched_pred_norm_u = matched_pred_norm / (
        matched_pred_norm.norm(dim=-1, keepdim=True) + 1e-8
    )

    # Cálculo da perda de orientação (1.0 - Cosine_Similarity)
    normal_loss_pred = 1.0 - (pred_norm_u * matched_tgt_norm_u).sum(-1)
    normal_loss_tgt = 1.0 - (matched_pred_norm_u * target_norm_u).sum(-1)

    cd_normal = normal_loss_pred.mean() + normal_loss_tgt.mean()

    # Combinar ambas com o peso estipulado
    return cd_xyz, (normal_weight * cd_normal)


def earth_movers_distance_sinkhorn(
    pred, target, normal_weight=0.5, eps=0.01, max_iter=100
):
    """
    Aproximação da Earth Mover's Distance (EMD) via Algoritmo de Sinkhorn.
    Força uma bijeção (mapeamento 1 para 1) entre as nuvens de pontos.

    pred, target: (B, N, 6) -> [X, Y, Z, Nx, Ny, Nz]
    eps: Parâmetro de regularização de entropia (valores menores = mais próximo da EMD exata)
    max_iter: Número máximo de iterações de Sinkhorn
    """
    B, N, _ = pred.shape
    device = pred.device

    pred_xyz = pred[..., :3]
    target_xyz = target[..., :3]
    pred_norm = pred[..., 3:]
    target_norm = target[..., 3:]

    # 1. Calcular a matriz de custo geométrico baseada em XYZ
    # (B, N, 1) + (B, 1, N) - 2 * (B, N, N)
    r_pred = torch.sum(pred_xyz**2, dim=-1, keepdim=True)
    r_tgt = torch.sum(target_xyz**2, dim=-1, keepdim=True)
    mul = torch.bmm(pred_xyz, target_xyz.transpose(1, 2))
    cost_matrix = r_pred - 2 * mul + r_tgt.transpose(1, 2)
    cost_matrix = torch.clamp(cost_matrix, min=0.0)  # Estabilidade numérica

    # 2. Algoritmo de Sinkhorn para encontrar a Matriz de Transporte Ótimo (P)
    # Matriz de Kernel (K)
    K = torch.exp(-cost_matrix / eps)

    # Inicializa os vetores de escala (distribuição uniforme de massa)
    u = torch.ones(B, N, dtype=pred.dtype, device=device) / N
    v = torch.ones(B, N, dtype=pred.dtype, device=device) / N

    # Loops de projeção alternada (Sinkhorn Knopp)
    for _ in range(max_iter):
        # Evita divisão por zero com micro epsilon
        u = 1.0 / (N * torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
        v = 1.0 / (
            N * torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-12
        )

    # P é a matriz de acoplamento ótimo que diz qual ponto predito vai para qual alvo
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)  # Shape: (B, N, N)

    # 3. Loss Geométrica (XYZ) baseada no acoplamento ótimo
    emd_xyz = torch.sum(P * cost_matrix, dim=(1, 2)).mean()

    # 4. Alinhamento das Normais usando os pesos de acoplamento de P
    # Normalização das normais originais para cosseno perfeito
    pred_norm_u = pred_norm / (pred_norm.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm_u = target_norm / (target_norm.norm(dim=-1, keepdim=True) + 1e-8)

    # Matriz de perda de cosseno entre todas as normais possíveis (B, N, N)
    # 1.0 - Cosine_Similarity
    normal_cost = 1.0 - torch.bmm(pred_norm_u, target_norm_u.transpose(1, 2))

    # Multiplica a perda das normais pela probabilidade de transporte P
    emd_normal = torch.sum(P * normal_cost, dim=(1, 2)).mean()

    # O Sinkhorn altera a escala da loss de coordenadas. Multiplicamos por N
    # para trazer a emd_xyz de volta à magnitude esperada das distâncias reais.
    emd_xyz_scaled = emd_xyz * N

    return emd_xyz_scaled, (normal_weight * emd_normal)
