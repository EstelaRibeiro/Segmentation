import cv2
import matplotlib.pyplot as plt
import numpy as np

from augmentation import get_train_augmentation


# ============================================================
# Paths
# ============================================================

IMAGE_PATH = r"d:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Processed Lung Segmentation\images\CHNCXR_0001_0.png"

MASK_PATH = r"d:\usuarios\estela.ribeiro\Meus Documentos\Datasets\rx_segmentacao\Processed Lung Segmentation\masks\CHNCXR_0001_0.png"


# ============================================================
# Read image and mask
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)


if image is None:
    raise ValueError(
        f"Could not read image: {IMAGE_PATH}"
    )

if mask is None:
    raise ValueError(
        f"Could not read mask: {MASK_PATH}"
    )


# ============================================================
# Check original mask
# ============================================================

print("Original image:")
print("Shape:", image.shape)
print("Min:", image.min())
print("Max:", image.max())

print("\nOriginal mask:")
print("Shape:", mask.shape)
print("Unique values:", np.unique(mask))


# ============================================================
# Augmentation
# ============================================================

transform = get_train_augmentation()


# ============================================================
# Generate several augmented versions
# ============================================================

N_AUGMENTATIONS = 6


fig, axes = plt.subplots(
    N_AUGMENTATIONS,
    4,
    figsize=(14, 4 * N_AUGMENTATIONS)
)


for i in range(N_AUGMENTATIONS):

    transformed = transform(
        image=image,
        mask=mask
    )

    aug_image = transformed["image"]
    aug_mask = transformed["mask"]


    # --------------------------------------------------------
    # Convert mask to binary for visualization
    # --------------------------------------------------------

    aug_mask_binary = (
        aug_mask > 0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[i, 0].imshow(
        image,
        cmap="gray"
    )

    axes[i, 0].set_title(
        "Original Image"
    )

    axes[i, 0].axis("off")


    # --------------------------------------------------------
    # Original mask
    # --------------------------------------------------------

    axes[i, 1].imshow(
        mask,
        cmap="gray"
    )

    axes[i, 1].set_title(
        "Original Mask"
    )

    axes[i, 1].axis("off")


    # --------------------------------------------------------
    # Augmented image + mask overlay
    # --------------------------------------------------------

    axes[i, 2].imshow(
        aug_image,
        cmap="gray"
    )

    axes[i, 2].imshow(
        aug_mask_binary,
        alpha=0.35
    )

    axes[i, 2].set_title(
        f"Augmented {i+1} + Mask"
    )

    axes[i, 2].axis("off")


    # --------------------------------------------------------
    # Augmented mask
    # --------------------------------------------------------

    axes[i, 3].imshow(
        aug_mask_binary,
        cmap="gray"
    )

    axes[i, 3].set_title(
        f"Augmented Mask {i+1}"
    )

    axes[i, 3].axis("off")


plt.tight_layout()

plt.show()