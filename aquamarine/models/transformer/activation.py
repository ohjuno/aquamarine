from typing import Callable, Optional, Union

import torch

from torch import Tensor
from torch.nn import Dropout, Linear, Module, Parameter
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.functional import relu, gelu

import aquamarine.models.transformer.functional as F


def _get_activation_fn(activation):
    if activation == "relu":
        return relu
    elif activation == "gelu":
        return gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)


class MultiHeadAttention(Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.dropout = dropout
        self.num_heads = num_heads
        self.dim_heads = embed_dim // num_heads
        assert self.dim_heads * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))

        self.bias = bias
        if bias:
            self.q_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.k_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.v_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
        else:
            self.register_parameter('q_proj_bias', None)
            self.register_parameter('k_proj_bias', None)
            self.register_parameter('v_proj_bias', None)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        if self.bias:
            constant_(self.q_proj_bias, 0.)
            constant_(self.k_proj_bias, 0.)
            constant_(self.v_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        attn_output = F.multi_head_attention(
            query, key, value, self.num_heads,
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
            self.q_proj_bias, self.k_proj_bias, self.v_proj_bias,
            self.out_proj.weight, self.out_proj.bias,
            dropout_p=self.dropout, training=self.training, attn_mask=attn_mask
        )
        return attn_output


class FeedForward(Module):

    def __init__(self, embed_dim, dim_feedforward=2048, dropout=0.,
                 activation: Union[str, Callable[[Tensor], Tensor]] = relu, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(FeedForward, self).__init__()
        self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation) if isinstance(activation, str) else activation

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
