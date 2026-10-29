# Estudo Comparativo de Modelos Supervisionados para Segmentação Pulmonar em Raio-X de Tórax

Este repositório contém um projeto de estudos aprofundado sobre **visão computacional aplicada à radiologia médica**, focado na tarefa de **segmentação semântica binária dos pulmões** a partir de radiografias de tórax (CXR).

O projeto implementa e compara, integralmente do zero em PyTorch, 6 grandes paradigmas arquiteturais da literatura: **FCN**, **U-Net**, **UNet++**, **Attention U-Net**, **DeepLabv3+** e **SegFormer (Vision Transformer)**.

---

## 1. Contexto e Dataset

O projeto foi construído utilizando os dados do conjunto público [Lung Segmentation from Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset), composto por radiografias obtidas a partir dos datasets de referência de Montgomery County (EUA) e Shenzhen Hospital (China).

### O Desafio Clínico
A segmentação precisa da silhueta pulmonar é a etapa precursora de múltiplos sistemas de diagnóstico auxiliado por computador (CADe/CADx) em pneumologia e radiologia, como:
- Cálculo automatizado do Índice Cardiotorácico (detecção de cardiomegalia).
- Detecção e quantificação de infiltrados por pneumonia, tuberculose e COVID-19.
- Avaliação de derrames pleurais e pneumotórax.

---

## 2. Estrutura do Repositório

```
supervised_models/
│
├── README.md                           # Guia geral e manual de estudos do projeto
│
├── docs/                               # Documentação didática e teórica detalhada
│   ├── 01_preparacao_dados.md          # Pipeline de dados, albumentations e DataLoader
│   ├── 02_funcoes_perda_metricas.md    # Formulação matemática de BCE, Dice e IoU
│   ├── 03_fcn.md                       # Fully Convolutional Network
│   ├── 04_unet.md                      # U-Net clássica e Skip Connections
│   ├── 05_unet_plus_plus.md            # UNet++ e vias aninhadas densas
│   ├── 06_attention_unet.md            # Attention U-Net e Attention Gates
│   ├── 07_deeplabv3_plus.md            # DeepLabv3+, Convoluções Atrous e ASPP
│   ├── 08_segformer.md                 # SegFormer e Vision Transformers (MiT)
│   └── 09_comparativo_modelos.md       # Síntese comparativa, memória e anatomia
│
├── models/                             # Implementações das arquiteturas neurais
│   ├── fcn.py                          # SimpleFCN
│   ├── unet.py                         # UNet
│   ├── unet_plus_plus.py               # UNetPlusPlus
│   ├── attention_unet.py               # AttentionUNet
│   ├── deeplabv3_plus.py               # DeepLabV3Plus
│   └── segformer.py                    # SegFormer
│
├── trained_models/                     # Checkpoints dos modelos treinados (.pth)
│
├── gerar_dataset_processado.py         # Script de pré-processamento offline (512x512)
├── dataset.py                          # Classe PyTorch LungDataset
├── data.py                             # Particionamento de dados e criação de DataLoaders
├── augmentation.py                     # Transformações de aumento de dados (Albumentations)
├── losses.py                           # Implementação do DiceLoss
├── metrics.py                          # Funções de cálculo do Dice Score e IoU Score
├── testes.py                           # Script de depuração e inspeção do Data Augmentation
│
├── train_fcn.py                        # Script de treino: FCN
├── train_unet.py                       # Script de treino: U-Net
├── train_unetpp.py                     # Script de treino: UNet++
├── train_attention_unet.py             # Script de treino: Attention U-Net
├── train_deeplabv3_plus.py             # Script de treino: DeepLabv3+
├── train_segformer.py                  # Script de treino: SegFormer
│
├── evaluate_fcn.py                     # Script de avaliação e inferência: FCN
├── evaluate_unet.py                    # Script de avaliação e inferência: U-Net
├── evaluate_unetpp.py                  # Script de avaliação e inferência: UNet++
├── evaluate_attention_unet.py          # Script de avaliação e inferência: Attention U-Net
├── evaluate_deeplabv3_plus.py          # Script de avaliação e inferência: DeepLabv3+
└── evaluate_segformer.py               # Script de avaliação e inferência: SegFormer
```

---

## 3. Requisitos e Instalação

Recomenda-se um ambiente Python 3.9+ com suporte a GPU (CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations scikit-learn matplotlib numpy
```

---

## 4. Guia de Execução Passo a Passo

### Passo 1: Pré-processamento dos Dados Brutos
O script [`gerar_dataset_processado.py`](/estudando_segmentacao/supervised_models/gerar_dataset_processado.py) padroniza o dataset bruto baixado do Kaggle.

1. Abra o arquivo [`gerar_dataset_processado.py`](/supervised_models/gerar_dataset_processado.py) e verifique se as variáveis de diretório apontam para os caminhos do seu disco:
   - `ROOT_DIR`: Pasta onde o dataset descompactado está localizado (contendo as subpastas `CXR_png` e `masks`).
   - `OUT_DIR`: Pasta onde as imagens processadas serão salvas (ex.: `Processed Lung Segmentation`).
2. Execute o script no terminal:
   ```bash
   python gerar_dataset_processado.py
   ```
3. O script irá:
   - Localizar e parear automaticamente as imagens com suas respectivas máscaras (removendo sufixos `_mask` quando necessário).
   - Redimensionar ambas para a resolução fixa de $512 \times 512$ pixels (`INTER_AREA` para a radiografia e `INTER_NEAREST` para a máscara).
   - Binarizar estritamente os rótulos ($0$ para fundo, $255$ para pulmão).
   - Gerar um relatório descritivo no terminal e exibir uma imagem de exemplo com overlay colorido.

---

### Passo 2: Testar e Visualizar o Aumento de Dados (*Data Augmentation*)
Antes de treinar os modelos, é recomendável inspecionar visualmente se as deformações afins e espelhamentos estão clinicamente coerentes e sincronizados entre imagem e máscara.

Execute o script [`testes.py`](/supervised_models/testes.py):
```bash
python testes.py
```
Isso abrirá uma janela com 6 variações aumentadas da mesma radiografia com sobreposição transparente da máscara, permitindo atestar a sincronia espacial do Albumentations configurado em [`augmentation.py`](/supervised_models/augmentation.py).

---

### Passo 3: Treinamento dos Modelos
Cada arquitetura possui seu próprio script de treino independente (`train_<modelo>.py`), já pré-configurado com hiperparâmetros apropriados:

| Modelo | Script de Treinamento | Batch Size | Épocas | Loss | Arquivo Salvo |
|---|---|---|---|---|---|
| **Simple FCN** | [`train_fcn.py`](f/estudando_segmentacao/supervised_models/train_fcn.py) | 8 | 10 | BCEWithLogits | `fcn_best_bce_dice_aug.pth` |
| **U-Net** | [`train_unet.py`](/supervised_models/train_unet.py) | 8 | 20 | BCE + Dice | `unet_best_bce_dice_aug.pth` |
| **UNet++** | [`train_unetpp.py`](/estudando_segmentacao/supervised_models/train_unetpp.py) | 8 | 10 | BCEWithLogits | `unetpp_best_aug.pth` |
| **Attention U-Net** | [`train_attention_unet.py`](/estudando_segmentacao/supervised_models/train_attention_unet.py) | 8 | 15 | BCE + Dice | `attention_unet_best_bce_dice_aug.pth` |
| **DeepLabv3+** | [`train_deeplabv3_plus.py`](/estudando_segmentacao/supervised_models/train_deeplabv3_plus.py) | 8 | 15 | BCE + Dice | `DeepLabV3Plus_best_bce_dice_aug.pth` |
| **SegFormer** | [`train_segformer.py`](/estudando_segmentacao/supervised_models/train_segformer.py) | 2 | 15 | BCE + Dice | `SegFormer_best_bce_dice_aug.pth` |

Exemplo de execução para treinar a U-Net:
```bash
python train_unet.py
```

Durante a execução:
- Os dados são particionados de forma determinística (`random_state=42`) em:
  - **Treino**: 80%
  - **Validação**: 10%
  - **Teste**: 10%
- A cada época, a perda média de treino e validação é exibida.
- Sempre que a perda de validação atinge um novo menor valor histórico, o modelo é salvo automaticamente com a indicação `← BEST`.

---

### Passo 4: Avaliação e Inspeção Visual nos Dados de Teste
Cada arquitetura conta com seu script dedicado de avaliação (`evaluate_<modelo>.py`), que:
1. Carrega o modelo com os melhores pesos salvos na pasta `trained_models/`.
2. Itera sobre as imagens do conjunto de teste (nunca vistas durante o treino).
3. Calcula as métricas quantitativas globais (**Dice Score** e **IoU / Jaccard Index**) com média e desvio padrão:
   $$\text{Dice: Média} \pm \text{Desvio Padrão} \quad | \quad \text{IoU: Média} \pm \text{Desvio Padrão}$$
4. Plota um grid visual detalhado contendo 8 amostras de teste com 3 linhas comparativas:
   - **Linha 1**: Radiografia de entrada (CXR).
   - **Linha 2**: Máscara manual de referência médica (Ground Truth).
   - **Linha 3**: Predição binária gerada pelo modelo (Threshold $> 0.5$).

Para avaliar a U-Net, execute:
```bash
python evaluate_unet.py
```
Para avaliar o SegFormer, execute:
```bash
python evaluate_segformer.py
```

---

## 5. Índice da Documentação Teórica e Didática

Para aprofundar seus estudos em cada técnica e método implementado, consulte os módulos didáticos dedicados na pasta `docs/`:

1. [**docs/01_preparacao_dados.md**](/estudando_segmentacao/supervised_models/docs/01_preparacao_dados.md)  
   *Pré-processamento offline, lógica de pareamento com sufixo `_mask`, classe `LungDataset`, normalização para ponto flutuante $[0, 1]$, particionamento sem vazamento de dados e transformações afins no Albumentations.*

2. [**docs/02_funcoes_perda_metricas.md**](/estudando_segmentacao/supervised_models/docs/02_funcoes_perda_metricas.md)  
   *Por que a acurácia de pixel falha em dados médicos desbalanceados, formulação matemática do `BCEWithLogitsLoss`, formulação contínua do `DiceLoss`, vantagens da função combinada `BCE + Dice`, métricas discretas `dice_score` e `iou_score` e relação analítica entre elas.*

3. [**docs/03_fcn.md**](/supervised_models/docs/03_fcn.md)  
   *Conceitos da Fully Convolutional Network, substituição de camadas densas por convoluções $1 \times 1$, superamostragem bilinear $4\times$, análise de parâmetros e por que a ausência de conexões de salto prejudica a nitidez de contornos pulmonares.*

4. [**docs/04_unet.md**](/estudando_segmentacao/supervised_models/docs/04_unet.md)  
   *A arquitetura U-Net clássica, caminho contrátil (análise de contexto) e expansivo (localização precisa), papel das Skip Connections via concatenação, blocos `DoubleConv` e convoluções transpostas.*

5. [**docs/05_unet_plus_plus.md**](/estudando_segmentacao/supervised_models/docs/05_unet_plus_plus.md)  
   *O problema do Semantic Gap entre codificador e decodificador, vias densas e aninhadas da UNet++, indexação matemática dos nós $X^{i,j}$ e acúmulo gradual de representações semânticas.*

6. [**docs/06_attention_unet.md**](/estudando_segmentacao/supervised_models/docs/06_attention_unet.md)  
   *Mecanismo de Atenção Aditiva nas conexões de salto (Attention Gates), formulação do sinal de gating $g$ e compatibilidade espacial, supressão de regiões fora do pulmão (ossos, abdômen) e estabilização com Batch Normalization.*

7. [**docs/07_deeplabv3_plus.md**](/estudando_segmentacao/supervised_models/docs/07_deeplabv3_plus.md)  
   *Convoluções Atrous (dilatadas) e expansão do campo receptivo sem perda de resolução, módulo ASPP com múltiplos atrous rates ($6, 12, 18$) e pooling global, decodificador com projeção de características de baixo nível para 48 canais.*

8. [**docs/08_segformer.md**](/estudando_segmentacao/supervised_models/docs/08_segformer.md)  
   *Transição das CNNs para Vision Transformers, Overlap Patch Embedding, Efficient Self-Attention com redução de complexidade via Spatial Reduction (`sr_ratio`), Mix-FFN com depthwise convolution para dispensar Positional Encodings e All-MLP Decoder.*

9. [**docs/09_comparativo_modelos.md**](/estudando_segmentacao/supervised_models/docs/09_comparativo_modelos.md)  
   *Quadro comparativo global de pesos, parâmetros e memória, evolução histórica dos paradigmas, regiões anatômicas críticas em CXR (ápices, seios costofrênicos, silhueta cardíaca) e recomendações para novos experimentos.*

---

## 6. Referências Bibliográficas dos Métodos

1. **FCN**: Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
2. **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.
3. **UNet++**: Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis (DLMIA)*.
4. **Attention U-Net**: Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *Medical Imaging with Deep Learning (MIDL)*.
5. **DeepLabv3+**: Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *European Conference on Computer Vision (ECCV)*.
6. **SegFormer**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *Advances in Neural Information Processing Systems (NeurIPS)*.
