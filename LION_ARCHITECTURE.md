# LION — Arquitetura Completa: VAE Hierárquico + Difusão Latente para Nuvens de Pontos 3D

---

## Slide 1 — O Problema

Gerar nuvens de pontos 3D de alta qualidade requer resolver duas tensões simultâneas:

- **Diversidade global** → formas diferentes (avião, cadeira, mesa) exigem espaços latentes bem separados
- **Coerência local** → cada um dos 2048 pontos precisa estar na superfície certa, sem buracos ou ruído

Abordagens diretas falham porque modelar 2048 × 3 = 6144 valores correlacionados de uma vez é intratável.

**Solução do LION**: decompor a forma em dois níveis de abstração e treinar um difusor hierárquico.

---

## Slide 2 — Visão Geral da Arquitetura

```
╔══════════════════════════════════════════════════════════════════════╗
║                         TREINAMENTO (2 fases)                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  FASE 1 — VAE                                                        ║
║  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           ║
║  │ nuvem bruta  │───▶│  GlobalEnc   │───▶│  z_g (256)   │           ║
║  │ (B, N, 6)    │    │  PVCNN       │    │  estilo global│           ║
║  │ xyz+normais  │    └──────────────┘    └──────┬───────┘           ║
║  │              │                               │                    ║
║  │              │    ┌──────────────┐    ┌──────▼───────┐           ║
║  │              │───▶│  LocalEnc    │───▶│  z_l (N,11)  │           ║
║  │              │    │  PVCNN+AdaGN │    │  pos+features │           ║
║  └──────────────┘    └──────────────┘    └──────┬───────┘           ║
║                                                  │                    ║
║                                         ┌────────▼───────┐          ║
║                                         │   Decoder       │          ║
║                                         │   PVCNN+AdaGN   │          ║
║                                         └────────┬───────┘          ║
║                                                  │                    ║
║                                         xyz_out (B, N, 3)            ║
║                                                                      ║
║  FASE 2 — DIFUSÃO (VAE congelado)                                    ║
║  ┌─────────────┐    ┌──────────────┐                                 ║
║  │ ruído N(0,I)│───▶│ StyleDenoiser│───▶ z_g gerado                 ║
║  │  (B, 256)   │    │  MLP 6 cam.  │                                 ║
║  └─────────────┘    └──────────────┘                                 ║
║                             │ condição                                ║
║  ┌─────────────┐    ┌───────▼──────┐                                 ║
║  │ ruído N(0,I)│───▶│PointDenoiser │───▶ z_l gerado                 ║
║  │  (B,N,11)   │    │  Conv1d 8c.  │                                 ║
║  └─────────────┘    └──────────────┘                                 ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Slide 3 — O Espaço Latente Hierárquico

### z_g — Latente Global (B, 256)

- Captura a **identidade e estilo global** da forma
- Vetor único por shape, igual a um VAE clássico
- Regularizado com KL → N(0,I) (`beta_style=1.0`)
- O difusor do Nível 1 gera z_g a partir de ruído Gaussiano puro

### z_l — Latente Local (B, N, 11)

Cada um dos N=2048 pontos tem **11 canais**:

```
z_l[:, :, :3]   ← 3 canais de POSIÇÃO    (onde o ponto está)
z_l[:, :, 3:]   ← 8 canais de FEATURE    (o que o ponto "significa")
```

- Canais de posição: ancorados às coordenadas reais do ponto (position anchor)
- Canais de feature: informação semântica por ponto, condicionada em z_g
- KL mínimo (`beta_pos_pts=0.0`, `beta_feat_pts=0.001`)
- O difusor do Nível 2 gera z_l condicionado no z_g gerado

---

## Slide 4 — PVCNN: O Bloco Fundamental

O PVCNN (Point-Voxel CNN) processa nuvens de pontos com dois ramos paralelos por bloco:

```
                    ┌─────────────────┐
                    │  N pontos       │
                    │  (B, N, C_in)   │
                    └────────┬────────┘
             ┌───────────────┴───────────────┐
             ▼                               ▼
    ┌─────────────────┐            ┌─────────────────┐
    │  VoxelBranch    │            │  PointBranch    │
    │                 │            │                 │
    │  discretiza em  │            │  MLP por ponto  │
    │  grade R³       │            │  (B, N, C_in)   │
    │  Conv3D         │            │  → (B, N, C/2)  │
    │  grid_sample    │            │                 │
    │  → (B, N, C/2)  │            └────────┬────────┘
    └────────┬────────┘                     │
             └───────────────┬───────────────┘
                             ▼
                    ┌─────────────────┐
                    │   concat        │
                    │  (B, N, C_out)  │
                    │  GroupNorm+GELU │
                    └─────────────────┘
```

- **VoxelBranch**: captura contexto volumétrico (relações entre regiões do espaço)
- **PointBranch**: captura features locais por ponto
- Juntos: precisão local + contexto global em cada bloco

**Quando condicionado** (LocalEncoder + Decoder), o GroupNorm é substituído por **AdaGN**:

```python
scale, shift = Linear(z_g).chunk(2)   # z_g controla a normalização
h = GroupNorm(features)
h = h * (1 + scale) + shift           # modulação adaptativa
```

Isso permite que z_g "direcione" como cada bloco processa os pontos locais.

---

## Slide 5 — O Position Anchor (Inovação Central)

### O Problema Sem Anchor

Sem o anchor, z_l é um latente abstrato. O decoder precisa "inventar" a posição de cada ponto a partir de valores aleatórios. Na epoch 0, os pontos decodificados são completamente aleatórios. O gradiente precisa ensinar as posições do zero — lento e instável.

```
epoch  0: CD = 1.30
epoch 10: CD = 0.72
epoch 25: CD = 0.26
epoch 50: CD = 0.13
```

### Com o Position Anchor (LION)

O LocalEncoder adiciona as coordenadas reais do ponto ao output de posição:

```python
# Dentro de LocalEncoder.forward():
mu_raw = fc_mu(features)                     # delta aprendido ≈ 0 no início
mu_l[:, :, :3] = mu_raw[:, :, :3] + xyz     # anchor: adiciona posição real
```

Na epoch 0: `mu_raw ≈ 0` (init com std=0.01), portanto `mu_l[:,:,:3] ≈ xyz`.

O decoder recebe z_l com posições reais desde o primeiro forward pass:

```python
# Dentro de LIONDecoder.forward():
xyz_init = tanh(z_l[:, :, :3])   # ≈ input_xyz desde epoch 0
delta    = output_head(features)  # ≈ 0 no início (init com std=0.01)
xyz_out  = delta + xyz_init       # skip: correção + referência
```

```
epoch  0: CD = 0.108   ← 12x melhor que sem anchor!
epoch 10: CD = 0.12
epoch 25: CD = 0.08
epoch 50: CD = 0.067   (ainda decrescendo)
```

### Por Que Funciona

Dois skips residuais encadeados constroem um caminho de gradiente direto:

```
Loss_recon → xyz_out → (delta + xyz_init) → features_pvcnn
                                  ↑
                             tanh(z_l[:,:,:3])
                                  ↑
                          mu_l[:,:,:3] = delta_enc + xyz
                                  ↑
                           input_xyz (dado real)
```

O encoder não precisa "descobrir" as posições — aprende apenas uma correção sobre elas.

---

## Slide 6 — Treinamento do VAE: Loss e Beta Annealing

### A Loss ELBO

```
Loss = Recon + beta(t) × [beta_pos  × KL_pos
                         + beta_feat × KL_feat
                         + beta_sty  × KL_sty]

onde:
  Recon      = Chamfer Distance(xyz_out, xyz_target)
  KL_pos     = KL(N(mu_l[:,:,:3],  σ) ‖ N(0,I))   ← canais de posição de z_l
  KL_feat    = KL(N(mu_l[:,:,3:],  σ) ‖ N(0,I))   ← canais de feature de z_l
  KL_sty     = KL(N(mu_g, σ_g) ‖ N(0,I))          ← z_g global
```

### Pesos KL (espelhando LION weight_kl_pt / weight_kl_feat / weight_kl_glb)

| Peso | Valor | Razão |
|---|---|---|
| `beta_pos_pts` | **0.0** | O anchor já fornece o sinal de posição; KL aqui luta contra a reconstrução |
| `beta_feat_pts` | **0.001** | Regularização mínima → difusor aprende melhor a distribuição |
| `beta_style` | **1.0** | z_g precisa ser Gaussiano para o difusor Nível 1 funcionar |

### Beta Annealing

```
beta(epoch) = 0.0  →  1.0   (linear em 150 epochs)
```

- **Epochs 0–50**: beta ≈ 0, treino puro de reconstrução
- **Epochs 50–150**: KL sobe gradualmente, encoder aprende a compactar
- **Epochs 150+**: beta = 1.0, espaço latente totalmente regularizado

Sem annealing, o KL alto no início faz o encoder colapsar para a prior antes de aprender qualquer coisa útil.

---

## Slide 7 — Treinamento dos DDPMs (Fase 2)

O VAE é **congelado**. Apenas os dois denoisers são treinados.

### Extração dos Latentes Limpos

```python
# Para cada batch de formas reais:
x_norm = normalize_pc(points)                    # (B, N, 6)
mu_g, _ = vae.global_encoder(x_norm)             # (B, 256) — sem ruído
mu_l, _ = vae.local_encoder(x_norm, mu_g)        # (B, N, 11) — anchor embutido
# mu_g e mu_l são os ALVOS para os DDPMs aprenderem a gerar
```

Usa-se a média posterior (mu), não a amostra ruidosa. Isso estabiliza o treinamento do difusor.

### DDPM — Forward Process (Adicionar Ruído)

```
z_t = √(ᾱ_t) · z_0  +  √(1 − ᾱ_t) · ε        ε ~ N(0, I)
```

Com o cosine schedule: `ᾱ_t` começa em ≈1 (t=0, nenhum ruído) e vai a ≈0 (t=T, ruído puro).

### DDPM — Loss (Predição de Ruído)

```python
# Nível 1 — StyleDenoiser:
t = randint(0, T)
z_g_t, ε = schedule.q_sample(mu_g, t)
ε_pred = style_denoiser(z_g_t, t, class_label)
loss_style = MSE(ε_pred, ε)

# Nível 2 — LatentPointDenoiser:
z_l_t, ε = schedule.q_sample(mu_l, t)
ε_pred = point_denoiser(z_l_t, t, z_g=mu_g)    # condicionado em z_g real
loss_point = MSE(ε_pred, ε)
```

Os dois DDPMs são treinados simultaneamente em cada batch — os dois otimizadores avançam em paralelo.

### StyleDenoiser (Nível 1)

```
z_g_t (B, 256)  +  t (embedding senoidal)  +  class_label (embedding aprendido)
    ↓
MLP Residual × 6 camadas
    (LayerNorm adaptativo: z_g_t é modulado pelo embedding de condição)
    ↓
ε_pred (B, 256)
```

Com **Classifier-Free Guidance**: durante treino, `cfg_dropout=0.1` substitui o label pelo token `uncond` — o modelo aprende tanto o condicional quanto o incondicional.

### LatentPointDenoiser (Nível 2)

```
z_l_t (B, N, 11)  +  t (embedding senoidal)  +  z_g (B, 256)
    ↓
Conv1d(11 → 256)   ← projeta cada ponto para espaço latente
    ↓
Conv1d ResBlock × 8  (GroupNorm adaptativo com cond = [t_emb | z_g])
    (processamento shared-across-points: todos os N pontos compartilham os pesos)
    ↓
Conv1d(256 → 11)   ← projeta de volta para espaço de ponto
    ↓
ε_pred (B, N, 11)
```

Cada ponto é processado independentemente, mas todos são condicionados no mesmo z_g. O denoiser aprende: "dado este estilo global e este nível de ruído, qual ruído foi adicionado a esta nuvem de pontos?"

---

## Slide 8 — Inferência: Geração de Novas Formas

### Passo 1 — Amostrar z_g (com Classifier-Free Guidance)

```
z_T ~ N(0, I)    ← ruído puro (B, 256)

para t = T, T-1, ..., 1, 0:
    ε_cond   = StyleDenoiser(z_t, t, class=cls)
    ε_uncond = StyleDenoiser(z_t, t, class=uncond_token)
    ε        = ε_uncond + guidance × (ε_cond − ε_uncond)   ← CFG
    z_{t-1}  = DDPM_step(z_t, ε, t)

z_g = z_0    (B, 256) — estilo global gerado
```

**guidance** controla qualidade vs diversidade:
- 1.0 → máxima diversidade (sem CFG)
- 3.0 → boa qualidade (recomendado)
- 5.0+ → amostras muito típicas da classe, menos variação

### Passo 2 — Amostrar z_l condicionado em z_g

```
z_l_T ~ N(0, I)    ← ruído puro (B, N, 11)

para t = T, T-1, ..., 1, 0:
    ε = LatentPointDenoiser(z_l_t, t, z_g=z_g)   ← z_g do passo 1
    z_l_{t-1} = DDPM_step(z_l_t, ε, t)

z_l = z_l_0    (B, N, 11) — nuvem de pontos no espaço latente
```

### Passo 3 — Decodificar com o VAE Congelado

```python
xyz_out, normals = vae.decoder(z_l, z_g)
# xyz_out: (B, 2048, 3) — coordenadas 3D normalizadas
# normals: (B, 2048, 3) — normais unitárias por ponto
```

O decoder VAE age como um **refinador aprendido**: recebe z_l (que imita a distribuição de mu_l do encoder) e produz a forma final limpa com suas correções residuais.

---

## Slide 9 — Por Que a Hierarquia Funciona

### Sem Hierarquia (difusão direta em xyz)

O difusor precisa gerar 2048 × 3 = 6144 valores correlacionados de uma vez. O modelo precisa simultaneamente decidir: "que tipo de forma?" e "onde cada um dos 2048 pontos está?". Difícil de treinar, lento para amostrar, difícil de condicionar em classe.

### Com a Hierarquia LION

```
Nível 1: "Esta forma é uma cadeira — estilo compacto, quatro pernas"
         z_g ~ StyleDenoiser(class="chair")    → (B, 256)
         (simples: espaço 256-dim Gaussiano, MLP pequeno)

         ↓ z_g carrega o "contexto global" para o nível 2

Nível 2: "Dado que é uma cadeira com esse estilo, gere onde ficam os 2048 pontos"
         z_l ~ PointDenoiser(z_g)    → (B, 2048, 11)
         (condicional: o denoiser sabe o "tema" antes de gerar a geometria)
```

O VAE fecha o loop: o decoder converte z_l gerado em xyz real, usando z_g para modular o refinamento.

---

## Slide 10 — Parâmetros de Treinamento Recomendados

### Fase 1 — VAE

```bash
python3 train_vae.py \
  --latent_dim 8 \        # 8 canais de feature por ponto (z_l total = 11)
  --beta_pos_pts 0.0 \    # sem KL nos canais de posição
  --beta_feat_pts 0.001 \ # KL mínimo nos canais de feature
  --beta_style 1.0 \      # KL pleno em z_g
  --beta_end 1.0 \        # beta máximo
  --beta_epochs 150 \     # annealing em 150 epochs
  --recon_loss chamfer \
  --epochs 500
```

### Fase 2 — Difusão

```bash
python3 train_diffusion.py \
  --vae_ckpt checkpoints/best.pt \   # lê latent_dim/style_dim do checkpoint
  --epochs 1000 \
  --T 1000 \                          # steps do DDPM
  --guidance 3.0 \                    # CFG para visualização
  --style_hidden 512 \                # capacidade do StyleDenoiser
  --style_layers 6 \
  --point_hidden 512 \                # capacidade do PointDenoiser
  --point_layers 8
```

---

## Slide 11 — Resumo do Fluxo de Dados

```
TREINAMENTO VAE
───────────────
pontos (B,N,6)
  ↓ normalize_pc
xyz_n + normals (B,N,6)
  ↓ GlobalEncoder [PVCNN × 3 + MaxPool + Linear]
(mu_g, logvar_g) → z_g (B, 256)
  ↓ LocalEncoder [PVCNN_AdaGN × 3 + Conv1d] + anchor skip
(mu_l, logvar_l) → z_l (B, N, 11)    mu_l[:,:,:3] ≈ input_xyz
  ↓ LIONDecoder [split + feat_proj + PVCNN_AdaGN × 3 + heads] + skip
xyz_out (B, N, 3)
  ↓
Loss = Chamfer(xyz_out, xyz_target) + beta × KL_separado

INFERÊNCIA (pós-treino)
───────────────────────
class_label ──▶ StyleDenoiser ──▶ z_g (B, 256)
                                      │
N(0,I) ──▶ PointDenoiser(· | z_g) ──▶ z_l (B, N, 11)
                                      │
                               Decoder(z_l, z_g)
                                      │
                               xyz_out (B, N, 3)  ✓
```

---

## Apêndice — Glossário

| Termo | Significado |
|---|---|
| **z_g** | Latente global (B, 256) — estilo da forma |
| **z_l** | Latente local (B, N, 11) — posição + feature por ponto |
| **Position Anchor** | Skip que inicializa z_l[:,:,:3] ≈ input_xyz |
| **PVCNN** | Point-Voxel CNN: dois ramos (voxel 3D + MLP por ponto) |
| **AdaGN** | Adaptive Group Norm: z_g modula a normalização via scale/shift |
| **KL_pos / KL_feat** | KL separado para canais de posição e feature de z_l |
| **beta annealing** | KL cresce de 0 a 1 em 150 epochs para evitar colapso |
| **CFG** | Classifier-Free Guidance: controla fidelidade à classe |
| **DDPM** | Denoising Diffusion Probabilistic Model |
| **ε-prediction** | O denoiser prediz o ruído adicionado, não x₀ diretamente |
| **free_bits** | Piso mínimo no KL por dimensão (evita colapso de dims individuais) |
