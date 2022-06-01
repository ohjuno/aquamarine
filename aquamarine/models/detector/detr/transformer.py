from typing import Callable, Optional, Union

import copy

from torch import Tensor
from torch.nn import Module, ModuleList, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.functional import relu

from aquamarine.models.transformer import MultiHeadAttention, FeedForward


def _get_clones(module: Module, n: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _with_positional_encoding(seq: Tensor, pos: Optional[Tensor] = None) -> Tensor:
    return seq if pos is None else seq + pos


class DETRTransformer(Module):

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETRTransformer, self).__init__()

        encoder_layer = DETREncoderLayer(embed_dim, num_heads, dim_feedforward, dropout,
                                         activation, layer_norm_eps, **factory_kwargs)
        encoder_norm = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = DETREncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = DETRDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout,
                                         activation, layer_norm_eps, **factory_kwargs)
        decoder_norm = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = DETRDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, pos: Optional[Tensor], query_pos: Optional[Tensor]) -> Tensor:
        r"""

        Args:
            src:
            tgt:
            pos:
            query_pos:

        Shape:
            ...
        """
        memory = self.encoder(src, pos=pos)
        output = self.decoder(tgt, memory, pos=pos, query_pos=query_pos)
        return output


class DETREncoder(Module):

    def __init__(self, encoder_layer: Module, num_layers: int, norm: Optional[Module] = None) -> None:
        super(DETREncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            pos: the positional encoding added to the queries and keys of the src sequence (optional).

        Shape:
            see the docs in DETR class.
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DETRDecoder(Module):

    def __init__( self, decoder_layer: Module, num_layers: int, norm: Optional[Module] = None) -> None:
        super(DETRDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layers in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the src sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            pos: the positional encoding from encoder (optional).
            query_pos: the positional encoding to form object queries (optional).

        Shape:
            see the docs in DETR class.
        """
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DETREncoderLayer(Module):

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.,
                 activation: Union[str, Callable[[Tensor], Tensor]] = relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETREncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.feed_forward = FeedForward(embed_dim, dim_feedforward, dropout, activation, **factory_kwargs)
        self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src:Tensor, src_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            pos:

        Shape:
            see the docs in DETR class.
        """
        x = src

        x = x + self._sa_block(self.norm1(x), src_mask, pos)
        x = x + self._ff_block(self.norm2(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        q = k = _with_positional_encoding(x, pos)
        x = self.self_attn(q, k, x, attn_mask=attn_mask)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.feed_forward(x)
        return self.dropout2(x)


class DETRDecoderLayer(Module):

    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.,
                 activation: Union[str, Callable[[Tensor], Tensor]] = relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETRDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.feed_forward = FeedForward(embed_dim, dim_feedforward, dropout, activation, **factory_kwargs)
        self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            pos:
            query_pos:

        Shape:
            see the docs in DETR class.
        """
        x = tgt

        x = x + self._sa_block(self.norm1(x), tgt_mask, query_pos)
        x = x + self._ma_block(self.norm2(x), memory, memory_mask, pos, query_pos)
        x = x + self._ff_block(self.norm3(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None) -> Tensor:
        q = k = _with_positional_encoding(x, query_pos)
        x = self.self_attn(q, k, x, attn_mask=attn_mask)
        return self.dropout1(x)

    # multihead attention block
    def _ma_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None) -> Tensor:
        q = _with_positional_encoding(x, query_pos)
        k = _with_positional_encoding(mem, pos)
        x = self.multihead_attn(q, k, mem, attn_mask=attn_mask)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.feed_forward(x)
        return self.dropout3(x)
