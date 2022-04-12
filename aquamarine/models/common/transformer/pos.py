from typing import Optional
import torch
import torch.nn as nn


class PESinusoidal(nn.Module):

    def __init__(
            self,
            in_features: int = 64,
            temperature: int = 10000,
            normalize: bool = False,
            scale: float = None,
            eps: float = 1e-6,
    ) -> None:
        super(PESinusoidal, self).__init__()
        self.in_features = in_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('Normalize should be True if scale is passed.')
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale
        self.eps = eps

    def forward(
            self,
            x,
            mask: Optional[torch.Tensor],
    ):
        assert mask is not None
        embedding_h = ~mask.cumsum(dim=1, dtype=torch.float32)
        embedding_w = ~mask.cumsum(dim=2, dtype=torch.float32)
        embedding_h = embedding_h / (embedding_h[:, -1:, :] + self.eps) * self.scale if self.normalize else embedding_h
        embedding_w = embedding_w / (embedding_w[:, :, -1:] + self.eps) * self.scale if self.normalize else embedding_w

        dimension_t = torch.arange(self.in_features, dtype=torch.float32, device=x.device)
        dimension_t = self.temperature ** (2 * (dimension_t // 2) / self.in_features)

        pos_h = embedding_h[:, :, :, None] / dimension_t
        pos_w = embedding_w[:, :, :, None] / dimension_t
        pos_h = torch.stack((pos_h[:, :, :, 0::2].sin(), pos_h[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_w = torch.stack((pos_w[:, :, :, 0::2].sin(), pos_w[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_h, pos_w), dim=3).permute(0, 3, 1, 2)
        return pos


class PELearned(nn.Module):

    def __init__(self, in_features: int = 256):
        super(PELearned, self).__init__()
        self.embedding_h = nn.Embedding(50, in_features)
        self.embedding_w = nn.Embedding(50, in_features)
        nn.init.uniform_(self.embedding_h)
        nn.init.uniform_(self.embedding_w)

    def forward(
            self,
            x,
    ):
        h, w = x.shape[-2:]
        embedding_h = self.embedding_h(torch.arange(w, device=x.device))
        embedding_w = self.embedding_w(torch.arange(h, device=x.device))
        pos = torch.cat([
            embedding_h.unsqueeze(0).repeat(h, 1, 1),
            embedding_w.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
