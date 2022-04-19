import torch
import torch.nn as nn

from einops import rearrange


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            dropout: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
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
