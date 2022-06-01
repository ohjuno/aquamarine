from typing import Iterable

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import aquamarine.datasets.coco.transforms as transforms

from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher
from aquamarine.modules import HungarianLoss


def get_parser():
    parser = argparse.ArgumentParser('DETR Trainer', add_help=False)

    # dataset and dataloader
    parser.add_argument('--root', default='/mnt/datasets/coco/train2017', type=str)
    parser.add_argument('--annFile', default='/mnt/datasets/coco/annotations/instances_train2017.json', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_worker', default=32, type=int)
    parser.add_argument('--resolution', default=640, type=int)

    # model
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--in_features', default=2048, type=int)

    # optimizer & criterion
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lamb_labels', default=1., type=float)
    parser.add_argument('--lamb_bboxes', default=5., type=float)
    parser.add_argument('--lamb_geniou', default=2., type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    # experiments
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=100, type=int)

    return parser


def main(args):

    factory_kwargs = dict(device=args.device, dtype=None)

    # dataloader
    transform = transforms.Compose([
        transforms.Resize(size=args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    trainset = COCODetection(root=args.root, annFile=args.annFile, transforms=transform)
    trainloader = COCODataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_worker, drop_last=True)

    # model
    backbone = nn.Sequential(*list(models.resnet50().children()))[:-2]
    detector = DETRTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    )
    model = DETR(
        backbone=backbone,
        transformer=detector,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        in_features=args.in_features,
    )
    model.to(args.device)

    # object queries & positional encodings
    object_queries = torch.zeros(args.batch_size, args.num_queries, 512, **factory_kwargs)
    pos = nn.Parameter(torch.empty((args.batch_size, int((args.resolution / 32) ** 2), 512), **factory_kwargs))
    query_pos = nn.Parameter(torch.empty((args.batch_size, args.num_queries, 512), **factory_kwargs))

    # optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    matcher = HungarianMatcher(
        lamb_labels=args.lamb_labels,
        lamb_bboxes=args.lamb_bboxes,
        lamb_geniou=args.lamb_geniou,
    )
    criterion = HungarianLoss(
        num_classes=args.num_classes,
        matcher=matcher,
        eos_coef=args.eos_coef,
    )
    criterion.to(args.device)

    # training loop
    for epoch in range(args.epochs):
        train(trainloader, model, object_queries, pos, query_pos, optimizer, criterion, epoch, args)


def train(
        dataloader: Iterable,
        model: nn.Module,
        object_queries: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        args,
):
    model.train()
    criterion.train()
    losses = []
    for idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = inputs.to(args.device)
        targets = [{k: v.to(args.device) for k, v in target.items()} for target in targets]
        outputs = model(inputs, object_queries, pos, query_pos)
        outputs = torch.split(outputs, [args.num_classes + 1, 4], dim=-1)
        outputs = dict(labels=outputs[0], bboxes=outputs[1])
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict[k] for k in loss_dict.keys())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f'\r'
            f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
            f'batch[{idx + 1:{len(str(len(dataloader)))}d}/{len(dataloader)}] - '
            f'loss(current batch): {loss.item():.3f} - '
            f'loss(accumulated): {np.nanmean(losses):.3f}',
            end='',
        )
    print(
        f'\r'
        f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
        f'average loss: {np.nanmean(losses):.3f}'
    )


if __name__ == '__main__':
    config = get_parser().parse_args()
    main(config)
