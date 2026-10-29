# Módulo 05: UNet++ (Nested and Dense Skip Pathways)

Arquivos relacionados:
- Arquitetura: [`models/unet_plus_plus.py`](/models/unet_plus_plus.py)
- Treinamento: [`train_unetpp.py`](/train_unetpp.py)
- Avaliação: [`evaluate_unetpp.py`](/evaluate_unetpp.py)

---

## 1. Contexto Teórico: O Problema do *Semantic Gap*

Na U-Net original, as conexões de salto conectam diretamente os mapas de características do codificador aos do decodificador na mesma resolução espacial. No entanto, existe uma disparidade semântica significativa entre eles (*Semantic Gap*):
- O codificador processa características de **baixo nível** (bordas, texturas primárias e ruídos de alta frequência), calculadas nas primeiras camadas da rede.
- O decodificador recebe características de **alto nível** (conceitos anatômicos e representações semânticas densas), que já atravessaram o gargalo (*bottleneck*).

A fusão direta dessas representações tão díspares pode dificultar a otimização pelo otimizador.

### A Solução da UNet++ (Zhou et al., DLMIA 2018)
A UNet++ reformula as conexões de salto substituindo os saltos diretos por **vias densas e aninhadas (*nested dense skip pathways*)**:
1. Constrói nós intermediários de convolução que transformam progressivamente as características do codificador antes de fundi-las com as do decodificador.
2. Cada nó intermediário recebe conexões residuais densas de todos os nós anteriores no mesmo nível espacial (inspirado nas DenseNets).
3. Reduz drasticamente a discrepância semântica antes da concatenação.

---

## 2. Diagrama da Topologia em Grade Aninhada

A nomenclatura matemática da UNet++ indexa cada nó como $X^{i,j}$, onde:
- $i$: Nível de subamostragem (profundidade espacial, onde $i=0$ é a resolução máxima $512 \times 512$).
- $j$: Índice do estágio aninhado ao longo da conexão de salto.

```
Nível 0 (512x512):   x00 -----------------> x01 -----------------> x02 (Saída Final)
                      |   \                  ^   \                  ^
                     pool  \____up10____/    |    \____up01____/
                      v                      |
Nível 1 (256x256):   x10 -----------------> x11 -----------------> x12
                      |   \                  ^   \                  ^
                     pool  \____up21____/    |    \____up12____/
                      v                      |
Nível 2 (128x128):   x20 -----------------> x21
                      |   \                  ^
                     pool  \____up32____/
                      v
Nível 3 (64x64):     x30 (Bottleneck)
```

---

## 3. Análise Detalhada dos Nós e Conexões

### 3.1. Coluna do Codificador ($j=0$)
As características são extraídas e subamostradas sequencialmente:
- $x_{00} = \text{DoubleConv}_{00}(X)$ $\to (32, 512, 512)$
- $x_{10} = \text{DoubleConv}_{10}(\text{Pool}(x_{00}))$ $\to (64, 256, 256)$
- $x_{20} = \text{DoubleConv}_{20}(\text{Pool}(x_{10}))$ $\to (128, 128, 128)$
- $x_{30} = \text{DoubleConv}_{30}(\text{Pool}(x_{20}))$ $\to (256, 64, 64)$

### 3.2. Primeiro Nível Aninhado ($j=1$)
Os nós da coluna $j=1$ recebem a concatenação do nó à esquerda no mesmo nível com o nó subjacente após a superamostragem:
- **Nó $x_{21}$**:
  $$x_{21} = \text{DoubleConv}_{21}( [x_{20}, \text{Up}_{32}(x_{30})] )$$
  - Entrada: $128 + 128 = 256$ canais $\to$ Saída: $128$ canais.
- **Nó $x_{11}$**:
  $$x_{11} = \text{DoubleConv}_{11}( [x_{10}, \text{Up}_{21}(x_{20})] )$$
  - Entrada: $64 + 64 = 128$ canais $\to$ Saída: $64$ canais.
- **Nó $x_{01}$**:
  $$x_{01} = \text{DoubleConv}_{01}( [x_{00}, \text{Up}_{10}(x_{10})] )$$
  - Entrada: $32 + 32 = 64$ canais $\to$ Saída: $32$ canais.

### 3.3. Segundo Nível Aninhado e Conectividade Densa ($j=2$)
Aqui entra a densidade completa: cada nó recebe **todos** os nós anteriores na mesma linha mais o nó de superamostragem da linha inferior:
- **Nó $x_{12}$**:
  $$x_{12} = \text{DoubleConv}_{12}( [x_{10}, x_{11}, \text{Up}_{12}(x_{21})] )$$
  - Como $x_{10}$, $x_{11}$ e $\text{Up}_{12}(x_{21})$ possuem cada um 64 canais:
  - Entrada: $64 + 64 + 64 = 192$ canais $\to$ Saída: $64$ canais.
- **Nó $x_{02}$ (Nó Decodificador Final)**:
  $$x_{02} = \text{DoubleConv}_{02}( [x_{00}, x_{01}, \text{Up}_{01}(x_{11})] )$$
  - Como $x_{00}$, $x_{01}$ e $\text{Up}_{01}(x_{11})$ possuem cada um 32 canais:
  - Entrada: $32 + 32 + 32 = 96$ canais $\to$ Saída: $32$ canais.

### 3.4. Projeção de Saída
```python
self.final = nn.Conv2d(32, 1, kernel_size=1)
return self.final(x02)
```

---

## 4. Treinamento e Avaliação

### Configuração do Treinamento: [`train_unetpp.py`](/train_unetpp.py)
- **Função de Custo**: `nn.BCEWithLogitsLoss()`.
- **Otimizador**: Adam com learning rate $10^{-3}$.
- **Batch Size**: 8.
- **Épocas**: 10 épocas.
- **Arquivo de Salvamento**: Salva o melhor modelo em `unetpp_best_aug.pth`.

### Avaliação: [`evaluate_unetpp.py`](/evaluate_unetpp.py)
- Avalia o checkpoint [`trained_models/unetpp_best.pth`](/trained_models/unetpp_best.pth) no conjunto de teste.
- Calcula médias e desvios padrão para Dice e IoU.
- Exibe o grid de visualização com predições binárias limiares ($> 0.5$).

---

## 5. Comparativo Teórico: U-Net vs. UNet++

| Característica | U-Net Padrão | UNet++ |
|---|---|---|
| **Conexões de Salto** | Diretas e rígidas | Aninhadas e densas com múltiplos caminhos |
| **Disparidade Semântica (*Semantic Gap*)** | Alta entre níveis correspondentes | Suavizada progressivamente pelos nós intermediários |
| **Reúso de Características** | Apenas uma vez por nível | Múltiplas vezes via concatenações densas |
| **Custo Computacional** | Menor memória e gradientes mais rápidos | Maior número de operações de convolução intermediárias |
