from typing import Optional

import copy
import torch
import torch.nn as nn

__all__ = [
    'TransformerEncoder',
    'TransformerDecoder',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
]


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def with_positional_encoding(t, pos:Optional[torch.Tensor]):
    return t if pos is None else t + pos


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            norm: nn.Module = None,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            x,
            x_mask: Optional[torch.Tensor] = None,
            x_key_padding_mask: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None,
    ):
        for layer in self.layers:
            x = layer(x, x_mask, x_key_padding_mask, pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(
            self,
            decoder_layer: nn.Module,
            num_layers: int,
            norm: nn.Module = None,
            return_intermediate: bool = False,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            x,
            memory,
            x_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            x_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
    ):
        intermediate = []
        for layer in self.layers:
            x = layer(
                x,
                memory,
                x_mask=x_mask,
                memory_mask=memory_mask,
                x_key_padding_mask=x_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(x))
        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.Sequential([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
            nn.Dropout(dropout),
        ])
        self.ff = nn.Sequential([
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        ])
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x,
            x_mask: Optional[torch.Tensor] = None,
            x_key_padding_mask: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None,
    ):
        n = self.norm_1(x)
        q = k = with_positional_encoding(n, pos)
        x = self.attn(q, k, value=n, attn_mask=x_mask, key_padding_mask=x_key_padding_mask)[0] + x
        n = self.norm_2(x)
        x = self.ff(n) + x
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.attn_1 = nn.Sequential([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
            nn.Dropout(dropout),
        ])
        self.attn_2 = nn.Sequential([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
            nn.Dropout(dropout),
        ])
        self.ff = nn.Sequential([
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        ])
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x,
            memory,
            x_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            x_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            pos: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
    ):
        n = self.norm_1(x)
        q = k = with_positional_encoding(n, pos)
        x = self.attn_1(q, k, value=n, attn_mask=x_mask, key_padding_mask=x_key_padding_mask)[0] + x
        n = self.norm_2(x)
        x = self.attn_2(
            query=with_positional_encoding(n, query_pos),
            key=with_positional_encoding(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0] + x
        n = self.norm_3(x)
        x = self.ff(n) + x
        return x
