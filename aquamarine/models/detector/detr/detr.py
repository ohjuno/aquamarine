from typing import Optional

import torch

from torch import Tensor
from torch.nn import Module, ModuleList, Conv2d, Linear
from torch.nn.functional import relu


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
        self.mlp_boxes = BoxFFN(embed_dim, embed_dim, 4, num_layers=3, **factory_kwargs)

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
        output = self.mlp_class(output), self.mlp_boxes(output).sigmoid()
        output = torch.cat(output, dim=-1)
        return output


class BoxFFN(Module):

    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(BoxFFN, self).__init__()
        hidden_dims = [hidden_features] * (num_layers - 1)
        self.layers = ModuleList(Linear(i, o, **factory_kwargs) for i, o in
                                 zip([in_features] + hidden_dims, hidden_dims + [out_features]))
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        r"""

        Args:
            x: ...
        """
        output = x
        for idx, layer in enumerate(self.layers):
            output = relu(layer(output)) if idx < self.num_layers - 1 else layer(output)
        return output
