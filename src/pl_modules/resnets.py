import torchvision
from torch import nn

def resnet18(pretrained=False, num_classes=None):
    assert num_classes is not None, "You must pass the number of classes to your model."
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model
