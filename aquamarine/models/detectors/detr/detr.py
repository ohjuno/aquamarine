import torch.nn as nn

from einops import rearrange


class DETR(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            transformer: nn.Module,
            num_classes: int,
            num_queries: int,
    ):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        embed_dim = self.transformer.embed_dim
        self.conv = nn.Conv2d(512, embed_dim, kernel_size=1, stride=1, bias=False)
        self.query_pos = nn.Embedding(num_queries, embed_dim).weight.unsqueeze(0)
        self.mlp_class = nn.Linear(embed_dim, self.num_classes + 1)
        self.mlp_boxes = nn.Linear(embed_dim, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.transformer(x, None, self.query_pos)
        c = self.mlp_class(x)
        b = self.mlp_boxes(x).sigmoid()
        return {'labels': c, 'bboxes': b}
