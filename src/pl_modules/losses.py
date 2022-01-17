import torch
from torch import nn
from torch.nn import functional as F


def no_loss(input, target, weights=None):
    """Dummy function."""
    return torch.tensor([0.])


def cce(input, target, weights=None):
    """Categorical crossentropy loss. Assumes input is logits."""
    if weights and weights is not None:
        weights_len = len(weights)
        weights = weights.reshape(1, weights_len, 1, 1)
    loss = nn.CrossEntropyLoss(weight=weights)
    target = torch.argmax(target, 1)
    output = loss(input, target)
    return output


def bce(input, target, weights=None):
    """Binary crossentropy loss. Assumes input is logits."""
    if weights and weights is not None:
        weights_len = len(weights)
        weights = weights.reshape(1, weights_len, 1, 1, 1)
    else:
        weights = None
    loss = nn.BCEWithLogitsLoss()  # weight=weights)

    # One hot the labels
    target_max = target.max()
    target = F.one_hot(
        target.to(torch.int64),
        int(target_max + 1)).to(
        torch.uint8)  # .permute(3, 0, 1, 2)
    output = loss(input, target.float())
    return output


def dice_loss(input, target, smooth=1., weights=None):
    """Dice loss. Assumes input is logits."""
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))  # noqa
