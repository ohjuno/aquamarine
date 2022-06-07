import torch
import aquamarine.models.backbone.volo.functional as F

from torch import Tensor
from torch.nn import Module, AvgPool2d, Linear, Parameter, Unfold
from torch.nn.init import constant_, xavier_uniform_


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)


class OutlookAttention(Module):

    def __init__(self, embed_dim, num_heads, kernel_size=3, stride=1,
                 bias=False, padding=1, dropout=0., device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(OutlookAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.dim_heads = embed_dim // num_heads
        assert self.dim_heads * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.v_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.a_proj_weight = Parameter(torch.empty((kernel_size ** 4 * num_heads, embed_dim), **factory_kwargs))

        self.bias = bias
        if bias:
            self.v_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.a_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
        else:
            self.register_parameter('v_proj_bias', None)
            self.register_parameter('a_proj_bias', None)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.unfold = Unfold(kernel_size, padding=padding, stride=stride)
        self.pool = AvgPool2d(stride, stride, ceil_mode=True)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.a_proj_weight)
        if self.bias:
            constant_(self.v_proj_bias, 0.)
            constant_(self.a_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, x: Tensor) -> Tensor:
        attn_output = F.outlook_attention(
            x, self.num_heads,
            self.v_proj_weight, self.a_proj_weight,
            self.v_proj_bias, self.a_proj_bias,
            self.out_proj.weight, self.out_proj.bias,
            kernel_size=self.kernel_size, stride=self.stride,
            unfold=self.unfold, pool=self.pool, dropout_p=self.dropout,
            padding=self.padding, training=self.training
        )
        return attn_output
