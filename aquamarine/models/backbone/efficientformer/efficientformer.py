from typing import *

import torch

from torch import Tensor
from torch.nn import Module, AdaptiveAvgPool2d, AvgPool2d, Linear, Sequential


class MetaBlock4D(Module):
    # MetaBlock4D implementation
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(self):
        super(MetaBlock4D, self).__init__()
        self.pool = AvgPool2d(kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        x = x + self.pool(x)



class MetaBlock3D(Module):
    # MetaBlock3D implementation
    # "EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>.

    expansion: int = 4

    def __init__(self):
        super(MetaBlock3D, self).__init__()

    def forward(self, x):
        pass


class EfficientFormer(Module):

    def __init__(self, num_classes):
        super(EfficientFormer, self).__init__()

        self.stem = Sequential()

        self.gap = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * 4, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # setup shape vars
        B, C, H, W = x.shape

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = x.contiguous().view(B, -1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientformer(pretrained: bool, **kwargs: Any):
    model = EfficientFormer()
    if pretrained:
        raise NotImplementedError
    return model


def efficientformer_l1(pretrained: bool = False, **kwargs: Any) -> EfficientFormer:
    r"""EfficientFormer-L1 model from
    `"EfficientFormer: Vision Transformers at MobileNet Speed" <https://arxiv.org/pdf/2206.01191.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _efficientformer(pretrained, **kwargs)
