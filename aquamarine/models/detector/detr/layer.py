from typing import Optional

import copy

from torch import Tensor
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.init import xavier_uniform_

from aquamarine.models.transformer import MultiHeadAttention, FeedForward


def _get_clones(module: Module, n: int):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _with_positional_encoding(seq: Tensor, pos: Optional[Tensor]):
    return seq if pos is None else seq + pos


class DETRTransformer(Module):

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETRTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_layer = DETREncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, **factory_kwargs)
        self.decoder_layer = DETRDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout, **factory_kwargs)
        encoder_norm = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        decoder_norm = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = DETREncoder(self.encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = DETRDecoder(self.decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, pos: Optional[Tensor], query_pos: Optional[Tensor]) -> Tensor:
        encoder_memory = self.encoder(src, pos)
        return self.decoder(tgt, encoder_memory, pos, query_pos)


class DETREncoder(Module):

    def __init__(self, encoder_layer: Module, num_layers: int, norm: Module = None) -> None:
        super(DETREncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, pos: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DETRDecoder(Module):

    def __init__( self, decoder_layer: Module, num_layers: int, norm: Module = None) -> None:
        super(DETRDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, encoder_memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_memory, pos, query_pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DETREncoderLayer(Module):

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETREncoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout, **factory_kwargs)
        self.norm_mhsa = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm_ff = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, pos: Optional[Tensor] = None) -> Tensor:
        q = k = _with_positional_encoding(self.norm_mhsa(x), pos)
        x = x + self.mhsa(q, k, self.norm_mhsa(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class DETRDecoderLayer(Module):

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETRDecoderLayer, self).__init__()
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout, **factory_kwargs)
        self.norm_mhsa = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm_mha = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm_ff = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, x, encoder_memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        q = k = _with_positional_encoding(self.norm_mhsa(x), query_pos)
        x = x + self.mhsa(q, k, self.norm_mhsa(x))
        q = _with_positional_encoding(self.norm_mha(x), query_pos)
        k = _with_positional_encoding(encoder_memory, pos)
        x = x + self.mha(q, k, encoder_memory)
        x = x + self.ff(self.norm_ff(x))
        return x
