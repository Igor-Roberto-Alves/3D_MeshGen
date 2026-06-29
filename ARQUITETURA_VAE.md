# Arquitetura do VAE Hierárquico para Nuvens de Pontos

Inspirado no paper **LION: Latent Point Diffusion Models for 3D Shape Generation** (NeurIPS 2022), porém com uma diferença central: **sem anchor bypass**. No LION original, as coordenadas xyz de entrada são passadas diretamente para o decoder como âncoras. Aqui, o decoder reconstrói posições inteiramente a partir do espaço latente estocástico — tornando o latente um prior generativo próprio para difusão posterior.

---

## Visão Geral

```
Entrada x (B, N, 6)  [xyz | normais]
         │
         ├──► GlobalEncoder ──► z_g  (B, style_dim)       latente global de forma
         │
         └──► LocalEncoder  ──► z_l  (B, N, latent_dim)   latente local por ponto
                  ▲
                  └── condicionado em z_g via AdaGN
                         │
                    LIONDecoder
                         │
                    xyz_out (B, N, 3)
```

O espaço latente tem **dois níveis**:

| Latente | Forma | Prior | Captura |
|---------|-------|-------|---------|
| `z_g` | `(B, style_dim)` | N(0, I) | Forma global — topologia, classe, proporções |
| `z_l` | `(B, N, latent_dim)` | N(0, I) | Variação local por ponto — posição e geometria local |

---

## Bloco PVCNN

Todos os estágios do encoder e decoder são baseados no **PVCNN** (Point-Voxel CNN), que combina dois branches em paralelo:

```
Entrada (B, N, C_in)
    │
    ├── VoxelBranch ──► voxeliza pontos em grid 3D ──► Conv3d ──► amostra de volta
    │                   captura contexto espacial local contínuo
    │
    └── PointBranch ──► SharedMLP ponto a ponto
                        captura features individuais sem discretização
    │
    └── Concatena → (B, N, C_out)
```

**VoxelBranch**: mapeia os pontos para um grid 3D de resolução R³, aplica Conv3d para propagar contexto espacial, e volta a amostrar as features nos pontos originais via `grid_sample`. Captura vizinhança local sem precisar de busca de vizinhos.

**PointBranch**: SharedMLP aplicada independentemente em cada ponto. Preserva detalhes finos que a discretização do voxel perderia (estruturas menores que 1 voxel).

---

## GlobalEncoder

**Arquivo**: `src/Encoder.py` → classe `GlobalEncoder`

Codifica a nuvem inteira em um único vetor de forma `z_g`.

```
x (B, N, 6)
    │
    ├── stage0: PVConvBlock(6→64,   res=32)
    ├── stage1: PVConvBlock(64→128, res=16)
    └── stage2: PVConvBlock(128→256, res=8)
         │
         └── Max-pool global → (B, 256)
              │
              ├── fc_mu     → mu_g    (B, style_dim)
              └── fc_logvar → logvar_g (B, style_dim)
```

A resolução do voxel decresce (32→16→8) à medida que as features ficam mais abstratas — hierarquia de receptive field crescente, igual a uma CNN convencional.

O vetor `z_g` é amostrado pela reparametrização:
```
z_g = mu_g + eps * exp(0.5 * logvar_g),  eps ~ N(0, I)
```

---

## LocalEncoder

**Arquivo**: `src/Encoder.py` → classe `LocalEncoder`

Codifica cada ponto individualmente, **condicionado em `z_g`** via AdaGN. O estilo global informa o encoder local sobre a classe e forma geral do objeto.

```
x (B, N, 6)  +  z_g (B, style_dim)
    │
    ├── stage0: PVConvBlockConditioned(6→64,    res=32, style=z_g)
    ├── stage1: PVConvBlockConditioned(64→128,  res=16, style=z_g)
    └── stage2: PVConvBlockConditioned(128→256, res=8,  style=z_g)
         │
         ├── Conv1d(256→latent_dim) → mu_l    (B, N, latent_dim)
         └── Conv1d(256→latent_dim) → logvar_l (B, N, latent_dim)
```

O bias do `fc_logvar` é inicializado em -6.0, forçando o encoder a começar quase determinístico (variância ≈ exp(-6) ≈ 0.002). O KL annealing abre gradualmente essa variância durante o treino.

**AdaGN (Adaptive Group Normalization)**: cada stage condicionado aplica GroupNorm padrão e depois escala e translada as features com parâmetros gerados dinamicamente a partir de `z_g`:
```
AdaGN(x, style):
    x_norm = GroupNorm(x)
    gamma, beta = Linear(style_dim → 2 * num_channels)
    return x_norm * (1 + gamma) + beta
```
O Linear é inicializado com pesos zero, então inicialmente o AdaGN age como identidade e aprende a modular progressivamente.

---

## LIONDecoder

**Arquivo**: `src/Decoder.py` → classe `LIONDecoder`

Reconstrói posições xyz a partir de `(z_l, z_g)` sem nenhuma informação da entrada original. O decoder opera em **três estágios com refinamento progressivo de posição**.

```
z_l (B, N, latent_dim)  +  z_g (B, style_dim)
    │
    ├── pos_head: Conv1d(latent_dim→64→3) + tanh
    │   └── xyz_cur (B, N, 3)   ← estimativa inicial de posição
    │
    ├── feat_proj: SharedMLP(latent_dim→128→256)
    │   └── feat (B, N, 256)
    │
    ├── stage0: PVConvBlockDecoder(256→256, res=16, style=z_g)
    │   └── refine0: Conv1d(256→3) → xyz_cur = tanh(xyz_cur + Δ)
    │
    ├── stage1: PVConvBlockDecoder(256→128, res=32, style=z_g)
    │   └── refine1: Conv1d(128→3) → xyz_cur = tanh(xyz_cur + Δ)
    │
    └── stage2: PVConvBlockDecoder(128→64, res=32, style=z_g)
         └── output_head: Conv1d(64→64→3)
              └── xyz_out = tanh(xyz_cur + Δ_final)
```

**Refinamento progressivo**: cada stage PVCNN voxeliza em `xyz_cur` atual. Após cada stage, `xyz_cur` é atualizado com uma correção residual. Isso significa que o grid de voxelização fica progressivamente mais preciso — o stage1 já vê uma grade melhor do que o stage0 viu.

**Saída residual**: a posição final é `tanh(xyz_cur + delta)` — o decoder aprende a *corrigir* uma estimativa já derivada do latente, em vez de prever posição absoluta do zero. Tarefa substancialmente mais fácil.

---

## Função de Loss — ELBO

**Arquivo**: `src/metric.py` → função `vae_loss`

```
L = L_recon + β · (KL_local + KL_global)
```

### Reconstrução

Suporta três modos:

| Modo | Descrição |
|------|-----------|
| `chamfer` | Chamfer Distance bidirecional com kNN (k=1) |
| `emd` | Earth Mover's Distance aproximada via Sinkhorn (50 iterações) |
| `both` | `(1 - w) · CD + w · EMD`, com `emd_weight` configurável |

**Chamfer** dá gradiente suave e é rápido. **EMD** força distribuição mais uniforme dos pontos (evita agrupamentos). Usar `both` combina os dois sinais.

### KL Divergence com Free Bits

```
KL[q(z|x) || N(0,I)] = -0.5 · (1 + logvar - mu² - exp(logvar))
```

Com **free bits** (threshold = 0.5): dimensões com KL < 0.5 não são penalizadas. Evita colapso posterior, onde o encoder ignora a entrada e o decoder tenta reconstruir de ruído puro.

### KL Annealing (β schedule)

β cresce linearmente de `beta_start` até `beta_end` ao longo de `beta_epochs`:

```
β(t) = beta_start + (t / beta_epochs) · (beta_end - beta_start)
```

Nas primeiras épocas β = 0, então a loss é puramente de reconstrução. O modelo aprende primeiro a reconstruir, e só depois é pressionado a manter o espaço latente próximo de N(0,I).

---

## Hiperparâmetros Principais

| Parâmetro | Valor padrão | Papel |
|-----------|-------------|-------|
| `latent_dim` | 3 | Dimensões do latente local por ponto |
| `style_dim` | 256 | Dimensões do latente global de forma |
| `in_channels` | 6 | xyz + normais por ponto |
| `beta_start` | 0.0 | β inicial (só reconstrução) |
| `beta_end` | 2.0 | β final |
| `beta_epochs` | 80 | Épocas até β máximo |
| `recon_loss` | `chamfer` | Tipo de loss de reconstrução |
| `lr` | 3e-4 | Learning rate (AdamW) |
| `warmup_epochs` | 10 | Warmup linear do LR |

---

## Fluxo Completo de Treino

```
1. Normaliza x para [-1, 1] (centrado, escala pelo max absoluto)
2. GlobalEncoder(x)       → mu_g, logvar_g
3. z_g = reparametrize(mu_g, logvar_g)
4. LocalEncoder(x, z_g)   → mu_l, logvar_l
5. z_l = reparametrize(mu_l, logvar_l)
6. LIONDecoder(z_l, z_g)  → xyz_out
7. L = chamfer(xyz_out, x_norm) + β · KL_local + β · KL_global
8. Backprop com grad clip (1.0) e AMP fp16
```

---

## Geração (Prior Sampling)

```python
z_g = randn(B, style_dim)           # amostrado do prior N(0, I)
z_l = randn(B, N, latent_dim)       # amostrado do prior N(0, I)
xyz = LIONDecoder(z_l, z_g)         # decodifica sem encoder
```

Como o espaço latente é treinado para se aproximar de N(0,I) pelo termo KL, amostras do prior produzem formas coerentes — base para o modelo de difusão posterior.
