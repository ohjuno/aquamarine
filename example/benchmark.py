from typing import Optional

import math
import torch

from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import dropout, linear, softmax, unfold

from torch.utils import benchmark


def scale_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tensor:
    B, Nt, E = q.shape
    scale = math.sqrt(E)
    q = q / scale
    # (B, Nt, E) @ (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn = attn + attn_mask if attn_mask is not None else attn
    attn = softmax(attn, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.)  # prevent nan resulting from padded-out inputs
    attn = dropout(attn, p=dropout_p) if dropout_p > 0.0 else attn
    # (B, Nt, Ns) @ (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output


def outlook_attention(
        x: Tensor,
        num_heads,
        v_proj_weight,
        kernel_size,
        stride,
        padding,
) -> Tensor:
    B, H, W, C = x.shape
    h, w = math.ceil(H / stride), math.ceil(W / stride)
    dim_heads = C // num_heads
    v = linear(x, v_proj_weight).permute(0, 3, 1, 2)  # (B, H, W, C) @ (..., C, N) -> (B, H, W, N) -> (B, N, H, W)
    v = unfold(v, kernel_size, padding=padding, stride=stride).view(B, num_heads, dim_heads, kernel_size ** 2, h * w)  # (B, H, W, N) -> (B, ..., HW) -> (B, H, N/H, kernel_size^2, HW)

    breakpoint()


if __name__ == '__main__':

    # query = key = value = torch.randn(1, 400, 512, device='cpu')
    #
    # t0 = benchmark.Timer(
    #     stmt='scale_dot_product_attention(q, k, v)',
    #     setup='from __main__ import scale_dot_product_attention',
    #     globals={'q': query, 'k': key, 'v':value},
    #     num_threads=10,
    # )
    #
    # print(t0.timeit(1000))

    embed_dim = 256
    feature = torch.randn(4, 14, 14, embed_dim)
    weights = Parameter(torch.empty((embed_dim, embed_dim)))

    oa = outlook_attention(feature, weights, 3, 1, 1)
    breakpoint()
