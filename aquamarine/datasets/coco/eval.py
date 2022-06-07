from typing import Dict

import torch

from torch.nn import Module
from torch.nn.functional import softmax

from aquamarine.datasets.utils import box_cxcywh_to_xyxy


class PostProcessing(Module):
    r"""Converts the outputs of the model into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        outputs = torch.split(outputs, [self.num_classes + 1, 4], dim=-1)
        outputs = dict(labels=outputs[0], bboxes=outputs[1])  # output format for onnx support
        out_labels, out_bboxes = outputs['labels'], outputs['bboxes']

        assert len(out_labels) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = softmax(out_labels, dim=-1)
        scores, labels = prob[..., :-1].max(-1)

        bboxes = box_cxcywh_to_xyxy(out_bboxes)
        h, w = target_sizes.unbind(1)
        scale = torch.stack([w, h, w, h], dim=-1)
        bboxes = bboxes * scale[:, None, :]

        results = [{'scores': s, 'labels': l, 'bboxes': b} for s, l, b in zip(scores, labels, bboxes)]
        return results
