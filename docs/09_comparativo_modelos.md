# Módulo 09: Comparativo Teórico e Síntese dos Modelos

Este documento consolida uma análise comparativa transversal entre todas as 6 arquiteturas implementadas no projeto, permitindo compreender a evolução dos paradigmas de segmentação semântica — desde as primeiras redes totalmente convolucionais até os Vision Transformers modernos.

---

## 1. Tabela Comparativa Geral

| Modelo | Família / Paradigma | Conexões de Salto (*Skip*) | Mecanismo Central | Perda no Treino | Tamanho do Modelo (*Checkpoint*) | Batch Size Típico |
|---|---|---|---|---|---|---|
| **Simple FCN** | CNN Clássica | Nenhuma | Convolução $1 \times 1$ + Upsampling Bilinear $4\times$ | BCEWithLogits | $\approx 264\text{ KB}$ ($\approx 66\text{K}$ parâmetros) | 8 |
| **U-Net** | Encoder-Decoder | Concatenação direta simples | Simetria contrátil-expansiva + DoubleConv | BCE + Dice | $\approx 7.7\text{ MB}$ ($\approx 1.9\text{M}$ parâmetros) | 8 |
| **UNet++** | Grade Aninhada Densa | Vias densas aninhadas ($X^{i,j}$) | Redução do *Semantic Gap* com nós intermediários | BCEWithLogits | $\approx 8.6\text{ MB}$ ($\approx 2.1\text{M}$ parâmetros) | 8 |
| **Attention U-Net** | Encoder-Decoder com Atenção | Portas de Atenção (*Attention Gates*) | Atenção aditiva $\psi$ suprime fundo anatômico | BCE + Dice | $\approx 125.7\text{ MB}$ ($\approx 31.4\text{M}$ parâmetros) | 8 |
| **DeepLabv3+** | Convoluções Dilatadas Multiescala | Salto de baixo nível projetado ($1\times1 \to 48$) | ASPP (Atrous Pyramid) + Pooling Global | BCE + Dice | $\approx 40.5\text{ MB}$ ($\approx 10.1\text{M}$ parâmetros) | 8 |
| **SegFormer** | Vision Transformer Hierárquico | Multiescala hierárquica (4 estágios) | Efficient Self-Attention (SR) + Mix-FFN + All-MLP | BCE + Dice | $\approx 13.8\text{ MB}$ ($\approx 3.4\text{M}$ parâmetros) | 2 |

---

## 2. A Evolução Conceitual das Arquiteturas

### Fase 1: O Surgimento da Segmentação Densa (FCN)
- **Marco**: Eliminar camadas densas (*fully connected*) permitindo predições com mapa bidimensional de saída.
- **Limitação no Raio-X**: Ao fazer o downsampling de $4\times$ sem guardar mapas intermediários, as bordas dos pulmões ficam serrilhadas ou excessivamente arredondadas.

### Fase 2: A Era das Conexões de Salto (U-Net)
- **Marco**: O caminho de expansão resgata as ativações de alta resolução do caminho de contração.
- **Ganho no Raio-X**: Recuperação nítida da linha pleural, dos ângulos costofrênicos e do contorno cardíaco.

### Fase 3: Refinamento Estrutural das Conexões (UNet++ e Attention U-Net)
- **UNet++**: Argumenta que a fusão de características com profundidades conceituais muito distintas gera ruído. Insere blocos convolucionais intermediários que transformam suavemente as características.
- **Attention U-Net**: Argumenta que passar toda a informação do encoder transfere também ruídos fora do pulmão (ex.: esterno, clavículas, ar circundante). O Attention Gate atua como máscara suave aprendida dinamicamente.

### Fase 4: Contexto Multiescala sem Perda de Resolução (DeepLabv3+)
- **Marco**: Convoluções dilatadas (*atrous*) expandem o campo de visão da rede sem precisar de dezenas de poolings. O ASPP permite enxergar o pulmão inteiro no ramo global e os vasos pulmonares nos ramos locais com taxas de dilatação variáveis ($r=6, 12, 18$).

### Fase 5: Atenção Global com Vision Transformers (SegFormer)
- **Marco**: Substitui filtros convolucionais fixos por auto-atenção adaptativa ao conteúdo. Consegue capturar relações entre o ápice pulmonar superior e a base diafragmática inferior simultaneamente, sem viés de localidade rígido.

---

## 3. Desafios Anatômicos Específicos em Raio-X de Tórax (CXR)

Durante o estudo e avaliação dos modelos nos arquivos `evaluate_*.py`, observe com atenção especial quatro regiões anatômicas críticas:

```
                  /\  Ápice Pulmonar (Alta sobreposição de clavículas)
                 /  \
                |    |  Hilo Pulmonar (Vasos calibrosos e brônquios)
                |    \____ Silhueta Cardíaca / Mediastino
                |         \
                 \________/  Ângulo Costofrênico (Terminação pontiaguda fina)
```

1. **Ápices Pulmonares**: Região superior onde as clavículas e primeiras costelas criam densidades ósseas sobrepostas que podem induzir falsos negativos. Modelos com portas de atenção (Attention U-Net) e Transformers tendem a suprimir melhor a densidade óssea da clavícula.
2. **Ângulos Costofrênicos**: Fendas agudas e afiladas na base dos pulmões. Modelos sem conexões de salto refinadas (como FCN) costumam arredondar erroneamente essa ponta.
3. **Borda Cardíaca e Mediastinal**: A transição de intensidade entre o ventrículo esquerdo do coração e o pulmão esquerdo é suave em imagens subexpostas. A função de perda composta `BCE + Dice` é essencial para evitar que a rede avance sobre o tecido cardíaco.
4. **Cúpula Diafragmática e Abdômen**: Presença eventual de bolha gástrica (ar no estômago abaixo do hemidiafragma esquerdo) pode ser confundida com parênquima pulmonar se o modelo não tiver contexto global suficiente (ponto forte do ASPP e SegFormer).

---

## 4. Recomendações Práticas para Novos Experimentos

1. **Se você dispõe de pouca memória VRAM ($\le 4\text{ GB}$)**:
   - A **U-Net** com batch size 8 ou o **SegFormer** com batch size 2 são ideais e convergem com alta fidelidade visual.
2. **Se o foco é máxima precisão em limites anatômicos finos**:
   - Compare a **Attention U-Net** com a **UNet++**. A presença dos Attention Gates fornece mapas de atenção interpretáveis do ponto de vista médico.
3. **Se você deseja explorar transfer learning futuro**:
   - O codificador MiT do **SegFormer** e o backbone do **DeepLabv3+** são amplamente suportados para inicialização com pesos pré-treinados em ImageNet, acelerando a convergência.
