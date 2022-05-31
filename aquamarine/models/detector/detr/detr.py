import torch

from torch.nn import Module, Conv2d, Embedding, Linear


class DETR(Module):

    def __init__(self, backbone: Module, transformer: Module,
                 num_classes: int, num_queries: int, in_features: int = 2048,
                 device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        embed_dim = self.transformer.embed_dim
        self.conv = Conv2d(in_features, embed_dim, kernel_size=1, stride=1, bias=False, **factory_kwargs)
        self.pos = Embedding(2500, embed_dim, **factory_kwargs).weight.unsqueeze(0)
        self.query_pos = Embedding(num_queries, embed_dim, **factory_kwargs).weight.unsqueeze(0)
        self.mlp_class = Linear(embed_dim, self.num_classes + 1, **factory_kwargs)
        self.mlp_boxes = Linear(embed_dim, 4, **factory_kwargs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        bsz, c, h, w = x.shape
        src = x.contiguous().view(bsz, h * w, c)
        bsz, ns, embed_dim = src.shape
        pos = self.pos[:, :ns, :].repeat(bsz, 1, 1)
        query_pos = self.query_pos.repeat(bsz, 1, 1)
        tgt = torch.zeros_like(query_pos)
        x = self.transformer(src, tgt, pos, query_pos)
        return {'labels': self.mlp_class(x), 'bboxes': self.box_cxcywh_to_xyxy(self.mlp_boxes(x).sigmoid())}

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
