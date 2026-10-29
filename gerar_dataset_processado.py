import os
import numpy as np
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# 1. Diretorios
ROOT_DIR = Path(r'd:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Lung Segmentation')

IMG_DIR = ROOT_DIR / 'CXR_png'
MASK_DIR = ROOT_DIR / 'masks'

OUT_DIR = Path(r'd:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Processed Lung Segmentation')

OUT_IMG_DIR = OUT_DIR / 'images'
OUT_MASK_DIR = OUT_DIR / 'masks'

# 2. Criar diretorios de saída
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# 3. Preprocessamento
def preprocessing_and_save(img_path, mask_path, out_img_dir, out_mask_dir, img_size=(512, 512)):
    """ Carrega uma CXR e sua máscara correspondente, redimensiona e salva
    """

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f'Não foi possível carregar a imagem: {img_path}')

    if mask is None:
        raise ValueError(f'Não foi possível carregar a máscara: {mask_path}')
    
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype('uint8')*255 # garantir mascara binária

    fig_name = Path(img_path).name

    out_img_path = out_img_dir / fig_name
    out_mask_path = out_mask_dir / fig_name

    cv2.imwrite(str(out_img_path), img)
    cv2.imwrite(str(out_mask_path), mask)

# 4. listar todas as imagens

all_images = sorted(IMG_DIR.glob('*.png'))
all_masks = sorted(MASK_DIR.glob('*.png'))

print(f'Total de imagens: {len(all_images)}')
print(f'Total de mascaras: {len(all_masks)}')

mask_dict = {}

for mask_path in all_masks:

    mask_name = mask_path.stem

    # Exemplo:
    #
    # CHNCXR_0001_0_mask
    #
    # vira:
    #
    # CHNCXR_0001_0

    if mask_name.endswith('_mask'):
        image_name = mask_name[:-5]
    else:
        image_name = mask_name

    mask_dict[image_name] = mask_path

# Parear imagens e máscaras
paired_data = []

missing_masks = []

for img_path in all_images:

    image_name = img_path.stem

    if image_name in mask_dict:

        mask_path = mask_dict[image_name]

        paired_data.append(
            (img_path, mask_path)
        )

    else:

        missing_masks.append(img_path)

print('\n==============================')
print('RELATÓRIO DO DATASET')
print('==============================')

print(f'Imagens encontradas:      {len(all_images)}')
print(f'Máscaras encontradas:     {len(all_masks)}')
print(f'Pares imagem/máscara:     {len(paired_data)}')
print(f'Imagens sem máscara:      {len(missing_masks)}')

# Processar pares
for img_path, mask_path in paired_data:

    preprocessing_and_save(
        img_path=img_path,
        mask_path=mask_path,
        out_img_dir=OUT_IMG_DIR,
        out_mask_dir=OUT_MASK_DIR,
        img_size=(512, 512)
    )


print('\nPreprocessing concluído!')
print(f'Imagens salvas em:   {OUT_IMG_DIR}')
print(f'Máscaras salvas em:  {OUT_MASK_DIR}')

def show_sample(img_path, mask_path):

    img = cv2.imread(
        str(img_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('CXR')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.3)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

img_path, mask_path = paired_data[0]

show_sample(img_path, mask_path)

mask = cv2.imread(
    str(paired_data[0][1]),
    cv2.IMREAD_GRAYSCALE
)

print(mask.shape)
print(mask.dtype)
print(mask.min())
print(mask.max())
print(np.unique(mask))