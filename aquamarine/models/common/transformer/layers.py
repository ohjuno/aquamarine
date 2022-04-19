import copy
import torch
import torch.nn as nn

from einops import rearrange


def _get_clones(obj, n):
    return nn.ModuleList([copy.deepcopy(obj) for _ in range(n)])


def with_positional_encoding(t, pos):
    return t if pos is None else t + pos


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout)
        self.project = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v):
        a = torch.einsum('b h n d, b h m d -> b h n m', q, k)
        a = self.softmax(a)
        a = self.dropout(a)
        x = torch.einsum('b h n m, b h m d -> b h n d', a, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.project(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):

    def __init__(
            self,
            dim: int,
            dim_feedforward: int,
            dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.feedforward = nn.Sequential(
            nn.Linear(self.dim, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, self.dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        x = self.feedforward(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int,
            dropout: float,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout)
        self.norm_mhsa = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x, pos):
        n = self.norm_mhsa(x)
        q = k = with_positional_encoding(n, pos)
        x = self.mhsa(q, k, n) + x
        n = self.norm_ff(x)
        x = self.ff(n) + x
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int,
            dropout: float,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout)
        self.norm_mhsa = nn.LayerNorm(embed_dim)
        self.norm_mha = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x, x_encoder, pos, query_pos):
        n = self.norm_mhsa(x)
        q = k = with_positional_encoding(n, pos)
        x = self.mhsa(q, k, n) + x
        n = self.norm_mha(x)
        q = with_positional_encoding(n, query_pos)
        k = with_positional_encoding(x_encoder, pos)
        x = self.mha(q, k, x_encoder) + x
        n = self.norm_ff(x)
        x = self.ff(n) + x
        return x


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
            pos=None,
    ):
        for layer in self.layers:
            x = layer(x, pos)
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
            x_encoder,
            pos=None,
            query_pos=None,
    ):
        intermediate = []
        for layer in self.layers:
            x = layer(
                x,
                x_encoder,
                pos,
                query_pos,
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
