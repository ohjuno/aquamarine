from typing import Optional

import torch

from torch import Tensor
from torch.nn import Module, Conv2d, Linear


class DETR(Module):

    def __init__(self, backbone: Module, transformer: Module,
                 num_classes: int, num_queries: int, in_features: int = 2048,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        embed_dim = self.transformer.embed_dim
        self.conv = Conv2d(in_features, embed_dim, kernel_size=1, stride=1, bias=False, **factory_kwargs)
        self.mlp_class = Linear(embed_dim, self.num_classes + 1, **factory_kwargs)
        self.mlp_boxes = Linear(embed_dim, 4, **factory_kwargs)

    def forward(self, x: Tensor, object_queries:Tensor,
                pos:Optional[Tensor] = None, query_pos:Optional[Tensor] = None) -> Tensor:
        r"""

        Args:
            ...

        Shape:
            ...
        """
        feature = x
        feature = self.backbone(feature)
        feature = self.conv(feature)

        bsz, c, h, w = feature.shape
        src = feature.contiguous().view(bsz, h * w, c)

        output = self.transformer(src, object_queries, pos, query_pos)
        return torch.cat((self.mlp_class(output), self.mlp_boxes(output).sigmoid()), dim=-1)
