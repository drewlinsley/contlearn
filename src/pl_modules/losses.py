import torch
from torch import nn


def cce(input, target):
    """Categorical crossentropy loss. Assumes input is logits."""
    loss = nn.CrossEntropyLoss()
    target = torch.argmax(target, 1)
    output = loss(input, target)
    return output


def bce(input, target):
    """Binary crossentropy loss. Assumes input is logits."""
    loss = nn.BCEWithLogitsLoss()
    output = loss(input, target)
    return output


def dice_loss(input, target, smooth=1.):
    """Dice loss. Assumes input is logits."""
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))  # noqa
