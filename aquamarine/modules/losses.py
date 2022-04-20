import torch
import torch.nn as nn
import torchvision.ops.boxes as bx


class BoxLoss(nn.Module):

    def __init__(self, lamb_l1: float = 1., lamb_giou: float = 1.):
        super(BoxLoss, self).__init__()
        self.lamb_l1 = lamb_l1
        self.lamb_giou = lamb_giou
        self.loss_l1 = torch.cdist
        self.loss_giou = bx.generalized_box_iou

    def forward(self, outputs, targets):
        pass


class MatchLoss(nn.Module):

    def __init__(self):
        super(MatchLoss, self).__init__()

    def forward(self, outputs, targets):
        pass


class HungarianLoss(nn.Module):

    def __init__(self):
        super(HungarianLoss, self).__init__()

    def forward(self, outputs, targets):
        pass
