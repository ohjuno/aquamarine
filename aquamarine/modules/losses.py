import torch
import torch.nn as nn


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
