import torch
import torch.nn as nn

from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import generalized_box_iou

from einops import rearrange


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):

    def __init__(self, lamb_labels: float = 1., lamb_bboxes: float = 1., lamb_geniou: float = 1.):
        super(HungarianMatcher, self).__init__()
        self.ll = lamb_labels
        self.lb = lamb_bboxes
        self.lg = lamb_geniou

    @torch.no_grad()
    def forward(self, outputs, targets):
        outputs_labels = rearrange(outputs['labels'], 'b n d -> (b n) d').softmax(dim=-1)
        outputs_bboxes = rearrange(outputs['bboxes'], 'b n d -> (b n) d')
        targets_labels = torch.cat([target['labels'] for target in targets])
        targets_bboxes = torch.cat([target['bboxes'] for target in targets])

        cost_labels = 1 - outputs_labels[:, targets_labels.long()]
        cost_bboxes = torch.cdist(outputs_bboxes, targets_bboxes, p=1)
        cost_geniou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(outputs_bboxes), targets_bboxes)
        cost = self.ll * cost_labels + self.lb * cost_bboxes + self.lg * cost_geniou
        cost = rearrange(cost, '(b n) d -> b n d', b=outputs['labels'].shape[0]).cpu()

        num_bboxes = [len(target['bboxes']) for target in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_bboxes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
