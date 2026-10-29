import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from data import create_datasets, create_dataloaders
from models.unet_plus_plus import UNetPlusPlus
from augmentation import get_train_augmentation


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    r'd:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Processed Lung Segmentation'
)

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

MODEL_PATH = "unetpp_best_aug.pth"


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
train_transform = get_train_augmentation()

train_dataset, val_dataset, test_dataset = create_datasets(
    IMAGE_DIR,
    MASK_DIR,
    test_size=0.10,
    val_size=0.10,
    random_state=42,
    train_transform=train_transform,
    val_transform=None,
    test_transform=None
)


# ============================================================
# DataLoaders
# ============================================================

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

model = UNetPlusPlus().to(device)

print("\nModel: UNET++")
print(model)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Best model
# ============================================================

best_val_loss = float("inf")


# ============================================================
# Training
# ============================================================

for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_running_loss += loss.item()


    val_loss = (
        val_running_loss /
        len(val_loader)
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        best_marker = " ← BEST"

    else:

        best_marker = ""


    # ========================================================
    # Print
    # ========================================================

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
        f"{best_marker}"
    )