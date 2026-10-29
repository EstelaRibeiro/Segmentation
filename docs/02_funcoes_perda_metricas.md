# Módulo 02: Funções de Perda e Métricas de Avaliação

Em tarefas de segmentação semântica em imagens médicas, a escolha das funções de perda (*loss functions*) e das métricas quantitativas é um dos fatores mais determinantes para o sucesso do treinamento. Este documento descreve as formulações matemáticas, o comportamento estatístico e a implementação em código dos módulos [`losses.py`](/losses.py) e [`metrics.py`](/metrics.py).

---

## 1. O Desafio do Desbalanceamento de Classes em Imagens Médicas

Em radiografias de tórax de $512 \times 512$ pixels ($262.144$ pixels no total):
- Os pulmões costumam ocupar entre $20\%$ e $35\%$ da área da imagem.
- O fundo anatômico (mediastino, costelas, diafragma, abdômen e ar externo) ocupa entre $65\%$ e $80\%$.

Se usarmos uma função de perda puramente baseada em acurácia de pixel (como a Cross Entropy padrão), o modelo pode convergir para mínimos locais em que prevê a maior parte dos pixels como fundo, alcançando aparentemente alta acurácia, mas falhando na demarcação dos limites dos órgãos.

---

## 2. Funções de Perda Implementadas: [`losses.py`](/losses.py)

### 2.1. Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
A entropia cruzada binária avalia cada pixel de forma independente como um problema de classificação binária:

$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$

Onde:
- $z_i \in \mathbb{R}$ é o logit não-normalizado emitido pelo modelo para o pixel $i$.
- $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ é a função logística Sigmoid.
- $y_i \in \{0, 1\}$ é o rótulo real (ground truth) do pixel $i$.

> **Vantagem Numérica do `BCEWithLogitsLoss`**: O PyTorch combina a ativação sigmoide e o cálculo da entropia cruzada em uma única camada usando o truque log-sum-exp, proporcionando maior estabilidade numérica e evitando *underflow* ou *overflow* de gradientes quando $\sigma(z) \to 0$ ou $\sigma(z) \to 1$.

---

### 2.2. Dice Loss: [`DiceLoss`](/losses.py#L5-L48)

O coeficiente de similaridade de Sørensen-Dice mede o grau de sobreposição entre duas regiões. Para tornar essa métrica diferenciável e apta à retropropagação (*backpropagation*), adota-se a aproximação contínua:

$$\mathcal{D}(P, Y) = \frac{2 \sum_{i} p_i y_i + \epsilon}{\sum_{i} p_i + \sum_{i} y_i + \epsilon}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \mathcal{D}(P, Y)$$

Onde:
- $p_i = \sigma(z_i) \in [0, 1]$ representa a probabilidade prevista para o pixel $i$.
- $y_i \in \{0, 1\}$ é a máscara alvo (ground truth).
- $\epsilon = 10^{-6}$ é o fator de suavização (*smooth*), evitando divisão por zero quando a máscara e a predição forem vazias.

#### Implementação no Código:
```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        loss = 1 - dice
        return loss.mean()
```

- **Operação por Amostra (`dim=1`)**: O achatamento (*flatten*) e as somas são calculados independentemente para cada imagem do batch, garantindo que o Dice de uma imagem não interfira no gradiente de outra antes da média final (`loss.mean()`).

---

### 2.3. Perda Composta: BCE + Dice Loss
Na maioria dos modelos deste projeto (UNet, Attention UNet, DeepLabv3+, SegFormer), utiliza-se a combinação aditiva:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

#### Por que essa combinação é superior?
1. **Gradientes Suaves no Início**: No início do treinamento, quando os pesos são aleatórios, o Dice Loss pode apresentar gradientes instáveis devido a pequenas interseções. O BCE fornece gradientes suaves e regulares em cada pixel individual.
2. **Foco na Forma e Fronteiras**: Conforme o treino avança, o Dice Loss força o modelo a maximizar a sobreposição global dos pulmões e penaliza fortemente falsos positivos e falsos negativos nos contornos pleurais.

---

## 3. Métricas de Avaliação: [`metrics.py`](/metrics.py)

Diferente das funções de perda, as métricas de avaliação final operam sobre máscaras binárias discretas ($0$ ou $1$), obtidas após a aplicação do limiar (*threshold*) de decisão:

$$\hat{Y}_{\text{bin}} = (\sigma(Z) > 0.5)$$

### 3.1. Dice Score: [`dice_score`](/metrics.py#L3-L12)

$$\text{Dice} = \frac{2 \times |P \cap Y|}{|P| + |Y|} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$

```python
def dice_score(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    dice = ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return dice
```
- Matematicamente equivalente ao **F1-Score** a nível de pixel.
- Varia no intervalo $[0, 1]$, onde $1.0$ representa segmentação perfeita.

---

### 3.2. Intersection over Union (IoU) / Índice de Jaccard: [`iou_score`](/metrics.py#L14-L32)

$$\text{IoU} = \frac{|P \cap Y|}{|P \cup Y|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

```python
def iou_score(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
```

---

### 3.3. Relação Matemática entre Dice e IoU
Existe uma relação determinística direta entre ambas as métricas:

$$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}} \quad \iff \quad \text{Dice} = \frac{2 \times \text{IoU}}{1 + \text{IoU}}$$

- Como consequência dessa propriedade, tem-se sempre:
  $$\text{IoU} \le \text{Dice}$$
- Por exemplo:
  - Se $\text{Dice} = 0.9500$, o $\text{IoU} = \frac{0.95}{1.05} \approx 0.9048$.
  - Se $\text{Dice} = 0.8000$, o $\text{IoU} = \frac{0.80}{1.20} \approx 0.6667$.
- O IoU é, portanto, mais estrito e penaliza erros de borda com maior severidade percentual que o Dice.
