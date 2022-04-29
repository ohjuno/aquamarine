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


args = {
    # configuration - dataloader
    'root': '/mnt/datasets/coco/train2017',
    'annFile': '/mnt/datasets/coco/annotations/instances_train2017.json',
    'batch_size': 40,
    'num_worker': 20,  # set 0 when debugging
    # configuration - experiment
    'device': 'cuda:0',
    'epochs': 100,
    # configuration - model
    # DETRTransformer
    'embed_dim': 512,
    'num_heads': 8,
    'dim_feedforward': 2048,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    # DETR
    'num_classes': 100,
    'num_queries': 100,
    'in_features': 2048,  # set 512 when backbone is ResNet18
    # configuration - optimizer
    'lr': 1e-3,
    # configuration - criterion
    # HungarianMatcher
    'lamb_labels': 1.,
    'lamb_bboxes': 5.,
    'lamb_geniou': 2.,
    # HungarianLoss
    'eos_coef': 0.1,
}


def main():
    # dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    trainset = COCODetection(root=args['root'], annFile=args['annFile'], transform=transform)
    trainloader = COCODataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'])

    # model
    backbone = nn.Sequential(*list(models.resnet50().children()))[:-2]
    detector = DETRTransformer(
        embed_dim=args['embed_dim'],
        num_heads=args['num_heads'],
        dim_feedforward=args['dim_feedforward'],
        num_encoder_layers=args['num_encoder_layers'],
        num_decoder_layers=args['num_decoder_layers'],
    )
    model = DETR(
        backbone=backbone,
        transformer=detector,
        num_classes=args['num_classes'],
        num_queries=args['num_queries'],
        in_features=args['in_features'],
    )
    model.to(args['device'])

    # optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    matcher = HungarianMatcher(
        lamb_labels=args['lamb_labels'],
        lamb_bboxes=args['lamb_bboxes'],
        lamb_geniou=args['lamb_geniou']
    )
    criterion = HungarianLoss(
        num_classes=args['num_classes'],
        matcher=matcher,
        eos_coef=args['eos_coef']
    )
    criterion.to(args['device'])

    # training loop
    for epoch in range(args['epochs']):
        train(trainloader, model, optimizer, criterion, args['device'], epoch, args['epochs'])


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
            f'\r'
            f'Epoch[{epoch + 1:{len(str(epochs))}d}/{epochs}] - '
            f'batch[{idx + 1:{len(str(len(dataloader)))}d}/{len(dataloader)}] - '
            f'loss(current batch): {loss.item():.3f} - '
            f'loss(accumulated): {np.nanmean(losses):.3f}',
            end='',
        )
    print(
        f'\r'
        f'Epoch[{epoch + 1:{len(str(epochs))}d}/{epochs}] - '
        f'average loss: {np.nanmean(losses):.3f}'
    )


if __name__ == '__main__':
    main()
