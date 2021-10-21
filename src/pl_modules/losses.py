from torch import nn


def cce(input, target):
    """Categorical crossentropy loss. Assumes input is logits."""
    loss = nn.CrossEntropyLoss()
    output = loss(input, target)
    return output


def bce(input, target):
    """Binary crossentropy loss. Assumes input is logits."""
    loss = nn.BCELoss()
    m = nn.Sigmoid()
    output = loss(m(input), target)
    return output


def dice_loss(input, target):
    """Dice loss. Assumes input is logits."""
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))  # noqa
