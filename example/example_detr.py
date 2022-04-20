import torch
import torch.nn as nn
import torchvision.models as models

from aquamarine.models import DETR, DETRTransformer


if __name__ == '__main__':
    backbone = nn.Sequential(*list(models.resnet18().children()))[:-2]
    detector = DETRTransformer(512, 8, 2048, 6, 6)
    model = DETR(backbone, detector, 20, 100)

    dummies = torch.rand((4, 3, 224, 224))
    outputs = model(dummies)
    breakpoint()
