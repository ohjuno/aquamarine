import torch
import torch.nn as nn

from einops import rearrange
from aquamarine.models.common.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


class Transformer(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            dim_feedforward,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
    ):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(embed_dim))
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(embed_dim))

        self.reset_parameter()

    def reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos, query_pos):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x_encoder = self.encoder(x, pos)
        obj_queries = torch.zeros_like(query_pos)
        x = self.decoder(obj_queries, x_encoder, pos, query_pos)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x
