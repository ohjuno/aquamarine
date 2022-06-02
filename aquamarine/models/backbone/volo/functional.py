from typing import Optional

import math
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout, fold, linear, softmax


def outlook_attention(
        x: Tensor,
        num_heads: int,
        v_proj_weight: Tensor,
        a_proj_weight: Tensor,
        v_proj_bias: Optional[Tensor],
        a_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        kernel_size: int,
        stride: int,
        unfold: Module,
        pool: Module,
        dropout_p: float,
        padding: int,
        training: bool = True,
):
    B, C, H, W = x.shape
    scale = math.sqrt(C / num_heads)
    dropout_p = 0.0 if not training else dropout_p
    v = linear(x.permute(0, 2, 3, 1), v_proj_weight, v_proj_bias).permute(0, 3, 1, 2)
    h, w = math.ceil(H / stride), math.ceil(W / stride)
    v = unfold(v).contiguous().view(B, num_heads, C // num_heads, kernel_size ** 2, h * w).permute(0, 1, 4, 3, 2)  # B, H, N, KK, E
    attn = pool(x).permute(0, 3, 1, 2)
    attn = linear(attn, a_proj_weight, a_proj_bias)
    attn = attn.contiguous().view(B, h * w, num_heads, kernel_size ** 2, kernel_size ** 2).permute(0, 2, 1, 3, 4)  # B, H, N, KK, KK
    attn = attn / scale
    attn = softmax(attn, dim=-1)
    attn = dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v).permute(0, 1, 4, 3, 2)
    output = output.contiguous().view(B, C * kernel_size ** 2, h * w)
    output = fold(output, output_size=(H, W), kernel_size=kernel_size, padding=padding, stride=stride)
    output = linear(output.permute(0, 2, 3, 1), out_proj_weight, out_proj_bias)
    output = dropout(output, p=dropout_p)
    return output
