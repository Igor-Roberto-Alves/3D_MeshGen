# SetVAE — Explicação da Abordagem

**Referência:** Kim et al., "SetVAE: Learning Hierarchical Composition for Generative Modeling of Set-Structured Data", CVPR 2021.

---

## O que é o SetVAE?

SetVAE é um VAE hierárquico projetado para gerar **conjuntos de pontos** (point clouds), imagens MNIST como conjuntos de pixels, ou qualquer coleção não ordenada de elementos. Ao contrário de VAEs convencionais que mapeiam para um vetor latente único e plano, o SetVAE aprende uma **representação latente hierárquica**: múltiplos níveis de latentes em diferentes escalas, do global ao local.

A ideia central é usar **Inducing Points** — um conjunto de "tokens resumo" aprendidos de tamanho fixo M — para criar um bottleneck de informação em cada nível. Isso evita que o modelo "trapaceie" copiando informação de entrada diretamente para a saída.

---

## Por que é melhor que o LION?

O LION (e nossa implementação anterior `Real_latent`) tinha um problema: o skip de coordenadas `z_l[:,:,:3] = xyz + 0.1*delta` permitia que o decoder copiasse as coordenadas de entrada, tornando o latente global `z_g` irrelevante (colapso posterior). O diagnóstico `zg_ablation` mostrava `cd_ratio ≈ 1.0`, confirmando que trocar `z_g` por ruído não prejudicava a reconstrução.

O SetVAE resolve isso de três formas:
1. **Sem skip de coordenadas**: o decoder gera xyz do zero a partir dos latentes, sem acesso às coordenadas de entrada.
2. **Bottleneck por inducing points**: a informação flui apenas pelo gargalo de M vetores aprendidos, impedindo cópia direta.
3. **Inferência bidirecional**: o encoder (bottom-up) extrai features hierárquicas da nuvem de entrada; o decoder (top-down) começa de um seed GMM e refina progressivamente usando correções posteriores do encoder — um mecanismo variacional robusto.

---

## Diagrama do Fluxo

```
ENCODER (bottom-up) — determinístico
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: xyz ∈ (B, N, 3)
  │
  ▼ Linear(3 → D) + ElemMLP
feat ∈ (B, N, D)
  │
  ▼ ISAB(enc_inds[0]=32 inds) → feat, h0 ∈ (B, 32, D)
  ▼ ISAB(enc_inds[1]=16 inds) → feat, h1 ∈ (B, 16, D)
  ▼ ISAB(enc_inds[2]= 8 inds) → feat, h2 ∈ (B,  8, D)
  ▼ ISAB(enc_inds[3]= 4 inds) → feat, h3 ∈ (B,  4, D)
  ▼ ISAB(enc_inds[4]= 2 inds) → feat, h4 ∈ (B,  2, D)
  ▼ ISAB(enc_inds[5]= 1 inds) → feat, h5 ∈ (B,  1, D)
  ▼ ISAB(enc_inds[6]= 1 inds) → feat, h6 ∈ (B,  1, D)
                                        ↓ (reversed → bu[0..6])
                              [h6, h5, h4, h3, h2, h1, h0]

DECODER (top-down) — estocástico
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
InitialSet (GMM, 4 mixtures, Fibonacci sphere init)
  │
  ▼ o ∈ (B, N, D)  ← seed set
  │
  ▼ DecoderBlock(1 ind, cond_prior=False)  ← nível coarso
    ├── project(o) → h ∈ (B, 1, D)
    ├── prior(h) → mu_p, logvar_p  (livre parâmetro aprendido, não condicionado)
    ├── posterior(bu[0] + None) → delta_mu, delta_logvar
    ├── z = (mu_p + delta_mu) + (sigma_p * delta_sigma) * eps
    ├── KL computado
    └── broadcast(fc(z), o) → o ∈ (B, N, D)
  │
  ▼ DecoderBlock(1 ind, cond_prior=True)   ← refina global
  ▼ DecoderBlock(2 inds, cond_prior=True)
  ▼ DecoderBlock(4 inds, cond_prior=True)
  ▼ DecoderBlock(8 inds, cond_prior=True)
  ▼ DecoderBlock(16 inds, cond_prior=True)
  ▼ DecoderBlock(32 inds, cond_prior=True) ← refina local
  │
  ▼ ElemMLP + Linear(D → 3)
xyz_out ∈ (B, N, 3)
```

---

## Componentes-Chave

### MAB (Multi-head Attention Block)
`MAB(x, y)`: x atende a y. Residual + FFN + LayerNorm.

Dois modos:
- **Slot Attention** (`project`): softmax sobre as *queries* (dim=1), divide por soma da massa de atenção. Garante que cada inducing point atenda a regiões diferentes da entrada — sem colapso de atenção.
- **Standard** (`broadcast`): softmax sobre as *keys* (dim=2). Cada ponto de saída agrega informação dos inducing points ponderando por relevância.

### ISAB (Induced Set Attention Block)
Encoder block com M inducing points aprendidos `I ∈ (1, M, D)`:
1. `project(x)`: slot-att MAB(I, x) → h ∈ (B, M, D) — comprime N→M
2. `broadcast(h, x)`: standard MAB(x, h) → (B, N, D) — expande M→N

O ISAB é O(N·M) em vez de O(N²), viabilizando N=2048 sem custo quadrático.

### InitialSet (GMM Seed)
Gera o conjunto inicial para o decoder sem depender da entrada:
- Parâmetros: logits, mu, sig para K=4 mixtures
- Mu inicializado na esfera unitária via lattice de Fibonacci (pontos maximamente separados)
- Usa Gumbel-Softmax para seleção diferenciável de mixture por ponto
- Saída: (B, N, D)

### DecoderBlock (Stochastic ABL)
Bloco variacional central:
- **Prior**: se `cond_prior=False` (nível 0): parâmetro livre (mu_p, logvar_p) aprendido. Se `cond_prior=True`: ElemMLP(h_td) → (mu_p, logvar_p) — prior condicionado nos features top-down anteriores.
- **Posterior residual**: `posterior(bu_h + td_h)` → (delta_mu, delta_logvar). O posterior **corrige** o prior, não o substitui.
- **Amostragem**: `z = (mu_p + delta_mu) + (sigma_p · delta_sigma) · eps`
- **KL**: `-0.5 · (delta_logvar + 1 - delta_mu²/sigma_p² - delta_sigma²)`, somado sobre M·z_dim dims por sample.
- Broadcast: `fc(z)` → o ∈ (B, N, D) via standard MAB.

### Espaço Latente Hierárquico
7 níveis com z_scales = [1,1,2,4,8,16,32]:
- **Escala 1 (1 ind × 16 dims)**: latente global — identidade da forma
- **Escala 32 (32 inds × 16 dims)**: latentes locais — detalhes geométricos
- Total: (1+1+2+4+8+16+32) × 16 = **1024 dims** de latente distribuídos hierarquicamente

---

## Hiperparâmetros Utilizados

| Parâmetro | Original (chair) | Nosso |
|-----------|-----------------|-------|
| `z_scales` | [1,1,2,4,8,16,32] | [1,1,2,4,8,16,32] |
| `z_dim` | 16 | 16 |
| `hidden_dim` | 64 | 64 |
| `n_mixtures` | 4 | 4 |
| `init_dim` | 32 | 32 |
| `num_heads` | 4 | 4 |
| `kl_warmup_epochs` | 2000 | 50 |
| `epochs` | 8000 | 200 |
| `beta` | 1.0 | 1.0 |
| `lr` | 1e-3 | 1e-3 |
| `slot_att` | True | True |
| `ln` (LayerNorm) | True | True |

O `kl_warmup_epochs=50` foi reduzido proporcionalmente (2000/8000 × 200 ≈ 50) para manter a mesma fração de treinamento com peso KL crescente.

---

## O que Esperar

**Qualidade de geração**: O SetVAE produz nuvens de pontos mais diversas e sem artefatos de "cubo" que surgem quando o espaço latente colapsa. A hierarquia garante que tanto a forma global quanto os detalhes locais sejam representados.

**Curva de treino**: Nas primeiras ~50 épocas, o KL é penalizado levemente (warmup). Esperar loss de reconstrução decrescendo rapidamente. Após warmup, o KL sobe para valores moderados enquanto a qualidade de geração melhora.

**Geração incondicional**: `model.generate(n)` apenas roda o decoder com prior puro — GMM seed → decoder blocks com prior apenas (sem bottom-up). A qualidade depende de quanto o prior foi bem treinado pelo KL.

**Tempo de treino**: Com N=2048, B=16, hidden=64: ~2-4s/batch em GPU. 200 épocas com dataset pequeno (~few thousand shapes) leva algumas horas.

**Limitações**: O modelo é sensível à escolha de `kl_warmup_epochs`. KL muito cedo → posterior collapse. KL muito tarde → geração ruim por prior fraco.

---

## Integração com Difusão (Próxima Etapa)

Após treinar o SetVAE, a difusão operará no espaço latente hierárquico:

```
Latentes z = {z_0, z_1, ..., z_6}  ← extraídos do SetVAE encoder
         ↓
  Concatenar ou processar hierarquicamente (a definir)
         ↓
  Denoiser DDPM aprende p(z) a partir de amostras do encoder
         ↓
  Na geração: amostrar z ~ p(z) via DDPM → SetVAE decoder → xyz
```

Duas abordagens possíveis:
1. **Flatten hierárquico**: concatenar todos z_i em um vetor plano, treinar DDPM padrão. Simples mas ignora estrutura.
2. **Difusão hierárquica**: DDPM separado por nível (ou hierárquico), respeitando a estrutura de escalas.

A abordagem 1 é mais simples para começar e é suficiente se o espaço latente for bem estruturado.
