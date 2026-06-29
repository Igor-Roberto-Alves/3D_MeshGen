# LION — Latent Point Diffusion Models for 3D Shape Generation

> NeurIPS 2022 · NVIDIA Research  
> Zeng et al., "LION: Latent Point Diffusion Models for 3D Shape Generation"

---

## 1. O Problema

Gerar nuvens de pontos 3D diretamente com um modelo de difusão é caro e difícil:

- Uma nuvem tem **2048 pontos × 3 coordenadas = 6144 dimensões** de ruído independentes
- O modelo de difusão teria que aprender a geometria global (forma do objeto) e local (superfície) ao mesmo tempo
- Difusão direta no espaço de pontos exige modelos muito grandes e é lenta

**Solução do LION:** comprimir a nuvem em um espaço latente compacto com um VAE hierárquico, e então aplicar difusão nesse espaço latente muito menor.

---

## 2. Visão Geral da Arquitetura

O LION tem **dois estágios** treinados separadamente:

```
ESTÁGIO 1 — VAE Hierárquico
  Nuvem de pontos → [Encoder] → z_g (global) + z_l (local) → [Decoder] → Nuvem reconstruída

ESTÁGIO 2 — Difusão em Dois Níveis (VAE congelado)
  Ruído → [StyleDenoiser] → z_g → [LatentPointDenoiser] → z_l → [Decoder VAE] → Nuvem gerada
```

---

## 3. Estágio 1: VAE Hierárquico

### 3.1 O Espaço Latente em Dois Níveis

O VAE codifica cada nuvem em **dois latentes complementares**:

| Latente | Forma | Significado |
|---------|-------|-------------|
| `z_g` (global) | `(B, style_dim)` — ex. `(B, 128)` | **Estilo global** da forma: identidade da classe, proporções gerais |
| `z_l` (local) | `(B, N, 3 + latent_dim)` — ex. `(B, 2048, 6)` | **Geometria local** por ponto: posição + features de superfície |

`z_l` tem duas partes:
- `z_l[..., :3]` → posições dos pontos âncora (estrutura geométrica)
- `z_l[..., 3:]` → features latentes por ponto (detalhes de superfície)

### 3.2 Encoder Global (`GlobalEncoder`)

Recebe a nuvem inteira `(B, N, 6)` (XYZ + normais) e produz `z_g`:

```
Nuvem (B, N, 6)
    → PointNet++ SetAbstraction (hierarquia de agrupamentos)
    → max-pool sobre todos os pontos
    → MLP
    → μ_g, σ_g  ∈ R^{style_dim}
    → z_g ~ N(μ_g, σ_g²)
```

### 3.3 Encoder Local (`LocalEncoder`)

Recebe a nuvem + `z_g` e produz `z_l` por ponto:

```
Nuvem (B, N, 6) + z_g (B, 128)
    → PointNet++ U-Net (SetAbstraction + FeaturePropagation)
    → features por ponto (B, N, C)
    → z_g é injetado via AdaGN em cada bloco
    → μ_l, σ_l  ∈ R^{N × latent_dim}
    → z_l ~ N(μ_l, σ_l²)
```

O `z_g` condiciona o encoder local via **Adaptive Group Normalization (AdaGN)**:

```python
# AdaGN: normaliza os features e reescala com z_g
x = GroupNorm(x)
scale, shift = Linear(z_g).chunk(2, dim=-1)
x = x * (1 + scale) + shift
```

### 3.4 Position Skip (Detalhe Crítico — Apêndice D.1)

Antes de passar `z_l` ao decoder, as **posições** são reconstruídas com um atalho residual:

```python
z_l[..., :3] = xyz_âncora + skip_weight * δ[..., :3]
# skip_weight = 0.1 (LION oficial)
```

Onde `δ` é a saída do encoder (deslocamento aprendido). O `skip_weight` pequeno força o modelo a **preservar a estrutura da nuvem original** nas posições, aprendendo apenas pequenas correções. Sem isso, o decoder tende a ignorar as posições e colapsar para formas desordenadas.

### 3.5 Decoder (`LIONDecoder`)

Reconstrói a nuvem a partir de `z_l` e `z_g`:

```
z_l (B, N, 3+latent_dim) + z_g (B, 128)
    → PVCBlock × L  (processamento híbrido voxel + ponto)
    → SharedMLP condicionado em z_g via AdaGN
    → xyz (B, N, 3) reconstruída
    → normais (B, N, 3) opcionais
```

**PVCBlock** é o bloco central do decoder:
- **Branch voxel**: voxeliza os pontos, aplica Conv3D, devolve features para os pontos (captura contexto global/volumétrico)
- **Branch ponto**: aplica Conv1d diretamente (captura detalhes locais)
- **Fusão**: Squeeze-and-Excitation + AdaGN com `z_g`

### 3.6 Função de Perda do VAE

```
L_VAE = L_recon + β_g · KL(z_g) + β_l · KL(z_l)

L_recon = L_chamfer_posição + β_normal · L_normal
KL(z)   = max(KL_vanilla, free_bits)   # free bits evita colapso posterior
```

O **KL annealing** aumenta `β` gradualmente de `β_start` a `β_end` durante `β_epochs` épocas, permitindo que o modelo aprenda a reconstrução antes de ser forçado a regularizar.

---

## 4. Estágio 2: Difusão em Dois Níveis

Com o VAE congelado, treinam-se dois modelos de difusão em cascata:

### 4.1 DDPM com Schedule Cossenoidal

O DDPM define um processo de difusão forward que adiciona ruído gradualmente:

```
q(z_t | z_0) = N(√ᾱ_t · z_0,  (1 - ᾱ_t) · I)
```

O schedule cossenoidal define `ᾱ_t` de forma que a transição de `z_0` (dado real) para `z_T` (ruído puro) seja suave. `T = 1000` passos.

O modelo aprende a **prever o ruído** `ε` dado `z_t` e o timestep `t`.

### 4.2 Normalização dos Latentes (Essencial)

O DDPM assume `z_0 ~ N(0, I)`. Os latentes do VAE **não** têm essa distribuição:

- `z_g`: ||média|| ≈ 11.5, std ≈ 0.18 → muito fora de escala
- `z_l`: std das features ≈ 0.001–0.01 → quase zero

**Solução:** padronizar antes de treinar o DDPM e desnormalizar antes do decoder:

```python
# Antes de treinar:
z_g_norm = (z_g - μ_g) / σ_g      # μ_g, σ_g calculados sobre todo o dataset
z_l_norm = (z_l - μ_l) / σ_l

# Na geração (depois de amostrar):
z_g_real = z_g_norm * σ_g + μ_g
z_l_real = z_l_norm * σ_l + μ_l
```

Sem essa normalização, o style denoiser aprende a gerar `z_g` na distribuição errada → point denoiser recebe condicionamento fora de distribuição → nuvens dispersas/caóticas.

### 4.3 StyleDenoiser (Nível Global)

Modela `p(z_g | classe)`:

```
Entrada: z_g_t (B, style_dim), timestep t, class label c
    → Embedding de classe (com dropout para CFG)
    → Embedding de timestep (sinusoidal → MLP)
    → ResBlocks MLP com condicionamento em (t_emb + class_emb)
    → ε_pred  ∈ R^{style_dim}
```

**Classifier-Free Guidance (CFG):** durante o treino, a classe é descartada com probabilidade `cfg_dropout=0.1`, forçando o modelo a aprender também a distribuição incondicional. Na geração:

```
ε_guided = ε_uncond + guidance · (ε_cond - ε_uncond)
```

Um `guidance` alto (ex. 3.0) aumenta a fidelidade à classe mas reduz diversidade.

### 4.4 LatentPointDenoiser (Nível Local)

Modela `p(z_l | z_g)` — condicionado no `z_g` gerado pelo StyleDenoiser:

```
Entrada: z_l_t (B, N, total_z_dim), timestep t, z_g (B, style_dim)
    → z_g injetado via AdaGN em cada bloco
    → ResBlocks Conv1d (opera independentemente por ponto)
    → LinearAttention entre pontos (captura dependências globais)
    → ε_pred  ∈ R^{N × total_z_dim}
```

`total_z_dim = 3 + latent_dim` (posições + features).

### 4.5 Função de Perda da Difusão

Loss simples de predição de ruído (eps-prediction):

```
L_style = E[||ε - ε_θ_style(z_g_t, t, c)||²]
L_point = E[||ε - ε_θ_point(z_l_t, t, z_g)||²]
L_total = L_style + L_point
```

### 4.6 EMA (Exponential Moving Average)

Os pesos EMA são usados na geração (mais estáveis que os pesos brutos):

```python
d_t = min(base_decay, (1 + n) / (10 + n))   # warmup nos primeiros updates
ema_p = d_t * ema_p + (1 - d_t) * p
```

O warmup é crítico: com `decay=0.9999` fixo e apenas 100 updates, os pesos EMA são 99% inicialização aleatória → geração em formato de cubo.

---

## 5. Pipeline de Geração

```
1. Amostrar ruído:  z_g_T ~ N(0, I)

2. Desnoisar z_g (T → 0 passos):
   Para t = T, T-1, ..., 1:
       ε_pred = StyleDenoiser(z_g_t, t, class=c)  [CFG aplicado]
       z_g_{t-1} = DDPM_step(z_g_t, ε_pred, t)

3. Desnormalizar: z_g_real = z_g_0 * σ_g + μ_g

4. Amostrar ruído:  z_l_T ~ N(0, I)   shape=(N, total_z_dim)

5. Desnoisar z_l (T → 0 passos):
   Para t = T, T-1, ..., 1:
       ε_pred = LatentPointDenoiser(z_l_t, t, z_g=z_g_real)
       z_l_{t-1} = DDPM_step(z_l_t, ε_pred, t)

6. Desnormalizar: z_l_real = z_l_0 * σ_l + μ_l

7. Decodar: xyz = VAE.decoder(z_l_real, z_g_real)
```

---

## 6. Por Que Funciona: Hierarquia de Informação

| Nível | Controla | Modelado por |
|-------|----------|--------------|
| `z_g` | Identidade da classe, proporções globais, "estilo" | StyleDenoiser + CFG |
| `z_l[:3]` | Posições âncora dos pontos (esqueleto geométrico) | LatentPointDenoiser |
| `z_l[3:]` | Features de superfície por ponto | LatentPointDenoiser |
| Decoder | Refinamento final, normais | PVCBlock + SharedMLP |

A separação global/local permite ao CFG controlar a **identidade da classe** (via `z_g`) sem precisar re-amostrar todos os 2048 pontos com guidance. O point denoiser simplesmente condiciona em `z_g` e preenche os detalhes locais.

---

## 7. Métricas de Avaliação

| Métrica | Mede | Direção |
|---------|------|---------|
| **MMD** (Minimum Matching Distance) | Qualidade média — cada shape gerada tem uma referência próxima? | ↓ melhor |
| **COV** (Coverage) | Diversidade — quantas referências são cobertas? | ↑ melhor |
| **1-NNA** (1-NN Accuracy) | Fidelidade geral — gerado e real são indistinguíveis? | → 0.5 ideal |

Um modelo que memoriza o dataset tem MMD≈0, COV≈100%, 1-NNA≈0.5 (perfeito, mas inútil). Um bom modelo equilibra os três.

---

## 8. Resumo dos Componentes e Onde Vivem

```
src/
├── Vae.py          # GlobalEncoder, LocalEncoder, LIONDecoder, Vae
├── Decoder.py      # PVCBlock, SharedMLP (com AdaGN), FeaturePropagation
├── Diffusion.py    # CosineSchedule, StyleDenoiser, LatentPointDenoiser
├── utils.py        # LinearAttention, AdaGN, SetAbstraction
├── dataset.py      # Ds_point_sampled_already, Ds_point_model
└── metric.py       # chamfer_distance_knn, generation_metrics (MMD/COV/1-NNA)

train_vae.py        # Estágio 1
train_diffusion.py  # Estágio 2 (inclui compute_latent_stats)
generate.py         # Amostragem + métricas
```
