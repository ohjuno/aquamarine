import torch
import torch.nn as nn

from einops import rearrange


class FeedForward(nn.Module):

    def __init__(
            self,
            dim: int,
            dim_feedforward: int,
            dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


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
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        a = torch.einsum('b h n d, b h m d -> b h n m', q, k)
        a = self.softmax(a)
        a = self.dropout(a)
        x = torch.einsum('b h n m, b h m d -> b h n d', a, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.project(x)
        x = self.dropout(x)
        return x
