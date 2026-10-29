import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data import create_datasets, create_dataloaders
from models.fcn import SimpleFCN

from metrics import dice_score, iou_score

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    r'd:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Processed Lung Segmentation'
)

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

BATCH_SIZE = 8

MODEL_PATH = "trained_models/fcn_best_aug.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

train_dataset, val_dataset, test_dataset = create_datasets(
    IMAGE_DIR,
    MASK_DIR,
    test_size=0.10,
    val_size=0.10,
    random_state=42
)

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = SimpleFCN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Evaluation
# ============================================================

dice_values = []
iou_values = []

with torch.no_grad():
    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        # forward
        outputs = model(images)
        # sigmoid
        probabilities = torch.sigmoid(outputs)
        # threshold
        predictions = (probabilities > 0.5).float()

        # metrics for each image
        for i in range (images.size(0)):

            dice = dice_score(predictions[i], masks[i])
            iou = iou_score(predictions[i], masks[i])

            dice_values.append(dice.item())
            iou_values.append(iou.item())


mean_dice = np.mean(dice_values)
std_dice = np.std(dice_values)

mean_iou = np.mean(iou_values)
std_iou = np.std(iou_values)


print("\n==============================")
print("FCN AUG TEST RESULTS")
print("==============================")

print(
    f"Number of test images: "
    f"{len(dice_values)}"
)

print(
    f"Dice: {mean_dice:.4f} ± {std_dice:.4f}"
)

print(
    f"IoU:  {mean_iou:.4f} ± {std_iou:.4f}"
)    


# ============================================================
# Get test batch
# ============================================================

images, masks = next(iter(test_loader))

images = images.to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    outputs = model(images)

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities > 0.5
    ).float()


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(
    3,
    8,
    figsize=(20, 8)
)

for i in range(8):

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    axes[0, i].imshow(
        images[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[0, i].set_title(f"CXR {i}")
    axes[0, i].axis("off")


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1, i].imshow(
        masks[i].squeeze(),
        cmap="gray"
    )

    axes[1, i].set_title("Ground Truth")
    axes[1, i].axis("off")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[2, i].imshow(
        predictions[i].cpu().squeeze(),
        cmap="gray"
    )

    axes[2, i].set_title("Prediction")
    axes[2, i].axis("off")


plt.tight_layout()
plt.show()