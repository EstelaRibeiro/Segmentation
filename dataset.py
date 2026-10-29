from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LungDataset(Dataset):

    def __init__(self, pairs, transform=None):

        self.pairs = pairs
        self.transform = transform

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        # --------------------------------------------------
        # Read mask
        # --------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Could not read mask: {mask_path}"
            )
        
        # --------------------------------------------------
        # Data augmentation
        # --------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # --------------------------------------------------
        # Normalize image
        # --------------------------------------------------

        image = image.astype(np.float32) / 255.0

        # --------------------------------------------------
        # Convert mask to binary 0/1
        # --------------------------------------------------

        mask = (mask > 0).astype(np.float32)

        # --------------------------------------------------
        # Add channel dimension
        # --------------------------------------------------

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # --------------------------------------------------
        # Convert to Tensor
        # --------------------------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask