from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher
from aquamarine.modules import HungarianLoss

# configuration
root = '/mnt/datasets/coco/train2017'
annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
batch_size = 100
device = 'cuda:0'
epochs = 100


def main():
    # dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    trainset = COCODetection(root=root, annFile=annFile, transform=transform)
    trainloader = COCODataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # model
    backbone = nn.Sequential(*list(models.resnet18().children()))[:-2]
    detector = DETRTransformer(512, 8, 2048, 6, 6)
    model = DETR(backbone, detector, num_classes=100, num_queries=100)
    model.to(device)

    # optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    matcher = HungarianMatcher(lamb_labels=1., lamb_bboxes=5., lamb_geniou=2.)
    criterion = HungarianLoss(num_classes=100, matcher=matcher, eos_coef=0.1)
    criterion.to(device)

    # training loop
    for epoch in range(epochs):
        train(trainloader, model, optimizer, criterion, device, epoch, epochs)


def train(
        dataloader: Iterable,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        epochs: int,
):
    model.train()
    criterion.train()
    losses = []
    for idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        outputs = model(inputs)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict[k] for k in loss_dict.keys())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f'Epoch[{epoch:{len(str(epochs))}d}/{epochs}] - '
            f'batch[{idx:{len(str(len(dataloader)))}d}/{len(dataloader)}] - '
            f'loss: {loss.item():.3f}'
        )
    print(
        f'Epoch[{epoch:{len(str(epochs))}d}/{epochs}] - '
        f'average loss: {np.nanmean(losses)}'
    )


if __name__ == '__main__':
    main()
