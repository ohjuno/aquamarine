import torch
import torch.nn as nn


from aquamarine.models.common.transformer.layers import *


class Transformer(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            return_intermediate,
    ):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.encoders = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(embed_dim))
        self.decoders = TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(embed_dim),
                                           return_intermediate)
        self._reset_parameters()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            x,
            mask,
            pos_encoding,
            query_pos_encoding,
    ):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)
        query_pos_encoding = query_pos_encoding.unsqueeze(1).repeat(1, b, 1)
        mask = mask.flatten(1)
        g = torch.zeros_like(query_pos_encoding)
        memory = self.encoders(x, x_key_padding_mask=mask, pos=pos_encoding)
        x = self.decoders(g, memory, memory_key_padding_mask=mask, pos=pos_encoding, query_pos=query_pos_encoding)
        return x.transpose(1, 2), memory.permute(1, 2, 0).view(b, c, h, w)
