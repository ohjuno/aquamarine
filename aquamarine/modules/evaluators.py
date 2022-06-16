import torch

from torch.nn import Module


class Accuracy(Module):

    def __init__(self, topk, reduction='none'):
        super(Accuracy, self).__init__()
        self.topk = topk
        self.reduction = reduction

    @torch.no_grad()
    def forward(self, outputs, targets):
        maxk = max(self.topk)
        bsz = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_ad(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / bsz))
        return res
