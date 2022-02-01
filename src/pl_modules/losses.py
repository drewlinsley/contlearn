import torch
from torch import nn
from torch.nn import functional as F

from monai import losses as monai_losses


class dice_loss:
    def __init__(self, weights=None, out_channels=None):
        """Generalized monai loss. Assumes input is logits."""
        self.loss_fun = monai_losses.DiceLoss(
            softmax=True,
            to_onehot_y=True)

    def forward(self, input, target):
        output = self.loss_fun(input.float(), target.float())
        return output


class dice_loss_mask_background:
    def __init__(self, weights=None, out_channels=None):
        """Generalized monai loss. Assumes input is logits."""
        self.loss_fun = monai_losses.DiceLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False)

    def __call__(self, input, target):
        output = self.loss_fun(input.float(), target.float())
        return output


class cce_thresh:
    def __init__(self, weights=None, out_channels=None):
        """Categorical crossentropy loss. Assumes input is logits."""
        self.loss_fun = nn.CrossEntropyLoss(weight=weights, reduction="none")

    def __call__(self, input, target):
        output = self.loss_fun(input.float(), target.squeeze(1).long())
        mask = target > 0
        pos_vals = output[mask]
        thresh = torch.median(pos_vals)
        raise NotImplementedError("Cant figure this out")
        pos_vals[pos_vals < thresh] = 0  # Fudge factor to ignore 
        return output


class cce:
    def __init__(self, weights=None, out_channels=None):
        """Categorical crossentropy loss. Assumes input is logits."""
        self.loss_fun = nn.CrossEntropyLoss(weight=weights)

    def __call__(self, input, target):
        target = target.squeeze(1)
        output = self.loss_fun(input.float(), target.long())
        return output


class bce:
    def __init__(self, weights=None, out_channels=None):
        """Categorical crossentropy loss. Assumes input is logits."""
        if weights and weights is not None:
            weights_len = len(weights)
            self.weights = weights.reshape(1, weights_len, 1, 1, 1)
            self.loss_fun = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.weights = False
            self.loss_fun = nn.BCEWithLogitsLoss()
        self.out_channels = out_channels

    def __call__(self, input, target):
        if self.out_channels > 1:
            target = F.one_hot(
                target.to(torch.int64),
                self.out_channels).to(
                torch.uint8).permute(0, 4, 1, 2, 3)
        output = self.loss_fun(input, target.float())
        if self.weights:
            output = output * self.weights  # Weight the logits
        return output


def thresh_cce(input, target, maxval=None, weights=None):
    """Categorical crossentropy loss. 

    Only evaluate the supra-threshold losses

    Assumes input is logits."""
    # if weights and weights is not None:
    #     weights_len = len(weights)
    #     weights = weights.reshape(1, weights_len, 1, 1)
    loss = nn.CrossEntropyLoss(weight=weights, reduction="none")
    # target = torch.argmax(target, 1)
    # output = loss(input.float(), target.float().squeeze(1))
    output = loss(input.float(), target.long().squeeze(1))

    # Grab + locations
    mask = target > 0
    pos_vals = output[mask]
    thresh = torch.median(pos_vals)
    pos_vals[pos_vals < thresh] = 0
    output[mask] = pos_vals
    return output
