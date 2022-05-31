from typing import Optional, Tuple

import math
import torch

from torch import Tensor
from torch.nn.functional import dropout, linear, softmax


def in_projection(
        q: Tensor, k: Tensor, v: Tensor,
        w_q: Tensor, w_k: Tensor, w_v: Tensor,
        b_q: Optional[Tensor] = None, b_k: Optional[Tensor] = None, b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


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
    attn = dropout(attn, p=dropout_p) if dropout_p > 0.0 else attn
    # (B, Nt, Ns) @ (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output


def memory_efficient_attention():
    # Self-Attention does not need O(n^2) Memory
    # https://arxiv.org/pdf/2112.05682.pdf
    pass


def multi_head_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        q_proj_weight: Tensor,
        k_proj_weight: Tensor,
        v_proj_weight: Tensor,
        q_proj_bias: Optional[Tensor],
        k_proj_bias: Optional[Tensor],
        v_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        dropout_p: float,
        training: bool = True,
        attn_mask: Optional[Tensor] = None,
):
    # set up shape vars
    bsz, nt, embed_dim = query.shape
    bsz, ns, embed_dim = key.shape
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        dim_heads = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        dim_heads = embed_dim // num_heads

    # in projection
    q, k, v = in_projection(
        query, key, value,
        q_proj_weight, k_proj_weight, v_proj_weight,
        q_proj_bias, k_proj_bias, v_proj_bias,
    )

    # reshape q, k, v for multi head attention
    q = q.contiguous().view(bsz * num_heads, nt, dim_heads)
    k = k.contiguous().view(bsz * num_heads, ns, dim_heads)
    v = v.contiguous().view(bsz * num_heads, ns, dim_heads)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # calculate attention
    attn_output = scale_dot_product_attention(q, k, v, attn_mask, dropout_p)

    # out projection
    attn_output = attn_output.contiguous().view(bsz, nt, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    return attn_output
