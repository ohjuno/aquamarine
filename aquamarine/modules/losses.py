from typing import Any, Dict

import torch
import torch.nn.functional as F

from torch.nn import Module
from torchvision.ops.boxes import generalized_box_iou
from aquamarine.datasets.utils.ops import box_cxcywh_to_xyxy


class HungarianLoss(Module):

    def __init__(self, num_classes: int, matcher: Any, weight_dict: Dict, eos_coef: float,
                 reduction: str = 'sum', device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super(HungarianLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.reduction = reduction

        empty_weight = torch.ones(self.num_classes + 1, **factory_kwargs)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @staticmethod
    def get_outputs_permutation_index(indices):
        batch_idx = torch.cat([torch.full_like(i, idx) for idx, (i, j) in enumerate(indices)])
        outputs_idx = torch.cat([i for (i, j) in indices])
        return batch_idx, outputs_idx

    @staticmethod
    def get_targets_permutation_index(indices):
        batch_idx = torch.cat([torch.full_like(j, idx) for idx, (i, j) in enumerate(indices)])
        targets_idx = torch.cat([j for (i, j) in indices])
        return batch_idx, targets_idx

    def get_loss_labels(self, outputs, targets, indices):
        indexes = self.get_outputs_permutation_index(indices)
        outputs_labels = outputs['labels']
        factory_kwargs = {'dtype': torch.int64, 'device': outputs_labels.device}
        targets_classes_matrix = torch.cat([target['labels'][j] for target, (i, j) in zip(targets, indices)])
        targets_classes = torch.full(outputs_labels.shape[:2], self.num_classes, **factory_kwargs)
        targets_classes[indexes] = targets_classes_matrix.type(torch.int64)
        return {'loss_labels': F.cross_entropy(outputs_labels.transpose(1, 2), targets_classes, self.empty_weight)}

    def get_loss_bboxes(self, outputs, targets, indices, num_bboxes):
        indexes = self.get_outputs_permutation_index(indices)
        outputs_bboxes = outputs['bboxes'][indexes]
        targets_bboxes = torch.cat([target['bboxes'][j] for target, (i, j) in zip(targets, indices)], dim=0)
        loss_bboxes = F.l1_loss(outputs_bboxes, targets_bboxes, reduction='none')
        loss_bboxes = loss_bboxes.sum() / num_bboxes
        loss_geniou = generalized_box_iou(box_cxcywh_to_xyxy(outputs_bboxes), box_cxcywh_to_xyxy(targets_bboxes))
        loss_geniou = 1 - torch.diag(loss_geniou)
        loss_geniou = loss_geniou.sum() / num_bboxes
        return {'loss_bboxes': loss_bboxes, 'loss_giou': loss_geniou}

    def forward(self, outputs, targets):
        outputs = torch.split(outputs, [self.num_classes + 1, 4], dim=-1)
        outputs = dict(labels=outputs[0], bboxes=outputs[1])  # output format for onnx support
        indices = self.matcher(outputs, targets)
        num_bboxes = sum(len(target["labels"]) for target in targets)
        loss_labels = self.get_loss_labels(outputs, targets, indices)
        loss_bboxes = self.get_loss_bboxes(outputs, targets, indices, num_bboxes)
        losses = {}
        losses.update(loss_labels)
        losses.update(loss_bboxes)
        output = sum(losses[k] for k in losses.keys()) if self.reduction == 'sum' else losses
        return output
