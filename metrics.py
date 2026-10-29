import torch

def dice_score(pred, target, smooth=1e-6):

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()

    dice = ( (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth) )

    return dice

def iou_score(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()

    union = (
        pred.sum()
        + target.sum()
        - intersection
    )

    iou = (
        (intersection + smooth)
        /
        (union + smooth)
    )

    return iou