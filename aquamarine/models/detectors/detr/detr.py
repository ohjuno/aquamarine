import torch.nn as nn

from einops import rearrange


class DETR(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            transformer: nn.Module,
            num_classes: int,
            num_queries: int,
            in_features: int = 512,
    ):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        embed_dim = self.transformer.embed_dim
        self.conv = nn.Conv2d(in_features, embed_dim, kernel_size=1, stride=1, bias=False)
        self.pos = nn.Embedding(2500, embed_dim).weight.unsqueeze(0)
        self.query_pos = nn.Embedding(num_queries, embed_dim).weight.unsqueeze(0)
        self.mlp_class = nn.Linear(embed_dim, self.num_classes + 1)
        self.mlp_boxes = nn.Linear(embed_dim, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        b, n, d = x.shape
        pos = self.pos[:, :n, :].repeat(b, 1, 1).to(x.device)
        query_pos = self.query_pos.repeat(b, 1, 1).to(x.device)
        x = self.transformer(x, pos, query_pos)
        return {'labels': self.mlp_class(x), 'bboxes': self.mlp_boxes(x).sigmoid()}
