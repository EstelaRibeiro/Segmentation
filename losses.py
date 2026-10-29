import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):

        super().__init__()

        self.smooth = smooth


    def forward(self, logits, targets):

        # Transform logits into probabilities
        probabilities = torch.sigmoid(logits)

        # Flatten each image
        probabilities = probabilities.view(
            probabilities.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        # Intersection
        intersection = (
            probabilities * targets
        ).sum(dim=1)

        # Dice
        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        # Dice Loss
        loss = 1 - dice

        return loss.mean()