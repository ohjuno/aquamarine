from typing import Optional
import copy
import torch
import torch.nn as nn

from aquamarine.models.common import FeedForward, MultiHeadAttention

__all__ = ['DETRTransformer']


def _get_clones(module: nn.Module, n: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def with_positional_encoding(seq: torch.Tensor, pos: Optional[torch.Tensor]):
    return seq if pos is None else seq + pos


class DETRTransformer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float = 0.1,
    ):
        super(DETRTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.encoder_layer = DETREncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.decoder_layer = DETRDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.encoder = DETREncoder(self.encoder_layer, num_encoder_layers, nn.LayerNorm(embed_dim))
        self.decoder = DETRDecoder(self.decoder_layer, num_decoder_layers, nn.LayerNorm(embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if 1 < p.dim():
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos: Optional[torch.Tensor], query_pos: Optional[torch.Tensor]):
        encoder_memory = self.encoder(x, pos)
        object_queries = torch.zeros_like(query_pos)
        return self.decoder(object_queries, encoder_memory, pos, query_pos)


class DETREncoder(nn.Module):

    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            norm: nn.Module = None,
    ):
        super(DETREncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, pos: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DETRDecoder(nn.Module):

    def __init__(
            self,
            decoder_layer: nn.Module,
            num_layers: int,
            norm: nn.Module = None,
    ):
        super(DETRDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, encoder_memory, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, encoder_memory, pos, query_pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DETREncoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int,
            dropout: float,
    ):
        super(DETREncoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout)
        self.norm_mhsa = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x, pos: Optional[torch.Tensor] = None):
        n = self.norm_mhsa(x)
        q = k = with_positional_encoding(n, pos)
        x = self.mhsa(q, k, n) + x
        n = self.norm_ff(x)
        x = self.ff(n) + x
        return x


class DETRDecoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dim_feedforward: int,
            dropout: float,
    ):
        super(DETRDecoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout)
        self.norm_mhsa = nn.LayerNorm(embed_dim)
        self.norm_mha = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_memory, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        n = self.norm_mhsa(x)
        q = k = with_positional_encoding(n, query_pos)
        x = self.mhsa(q, k, n) + x
        n = self.norm_mha(x)
        q = with_positional_encoding(n, query_pos)
        k = with_positional_encoding(encoder_memory, pos)
        x = self.mha(q, k, encoder_memory) + x
        n = self.norm_ff(x)
        x = self.ff(n) + x
        return x
