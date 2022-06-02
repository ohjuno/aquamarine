from typing import Any, Iterable, List

import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import aquamarine.datasets.coco.transforms as transforms

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from aquamarine.datasets import COCODetection, COCODataLoader
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher
from aquamarine.modules import HungarianLoss
from aquamarine.optimizer import MADGRAD, CosineAnnealingWarmRestarts
from aquamarine.utils import CKPT


def get_parser():
    parser = argparse.ArgumentParser('DETR Trainer', add_help=False)

    # dataset and dataloader
    parser.add_argument('--train_root', default='/mnt/datasets/coco/train2017', type=str)
    parser.add_argument('--train_annFile', default='/mnt/datasets/coco/annotations/instances_train2017.json', type=str)
    parser.add_argument('--valid_root', default='/mnt/datasets/coco/val2017', type=str)
    parser.add_argument('--valid_annFile', default='/mnt/datasets/coco/annotations/instances_val2017.json', type=str)
    parser.add_argument('--batch_size', default=380, type=int)
    parser.add_argument('--num_worker', default=16, type=int)
    parser.add_argument('--resolution', default=224, type=int)

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
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lamb_labels', default=1., type=float)
    parser.add_argument('--lamb_bboxes', default=5., type=float)
    parser.add_argument('--lamb_geniou', default=2., type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    # experiments
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--checkpoint', default='/tmp/runs/detr_r50_madgrad_scheduled_1', type=str)
    parser.add_argument('--tensorboard', default='/tmp/runs/detr_r50_madgrad_scheduled_1', type=str)
    parser.add_argument('--amp', default=True, type=bool)

    return parser


def main(args):
    factory_kwargs = dict(device=args.device, dtype=None)

    # dataloader
    transform = transforms.Compose([
        transforms.Resize(size=args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    trainset = COCODetection(root=args.train_root, annFile=args.train_annFile, transforms=transform)
    validset = COCODetection(root=args.valid_root, annFile=args.valid_annFile, transforms=transform)
    trainloader = COCODataLoader(trainset, args.batch_size, True, num_workers=args.num_worker, drop_last=True)
    validloader = COCODataLoader(validset, args.batch_size, False, num_workers=args.num_worker, drop_last=True)
    # for now, `drop_last` option should be set to True.
    # If this option is off, there is a possibility that the last batch and positional encoding will conflict

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
    # TODO: positional encoding that can support dynamic batch size and onnx at the same time is required

    # optimizer, criterion, scheduler and amp (automatic mixed precision)
    optimizer = MADGRAD(model.parameters(), lr=args.lr)
    matcher = HungarianMatcher(lamb_labels=args.lamb_labels, lamb_bboxes=args.lamb_bboxes, lamb_geniou=args.lamb_geniou)
    criterion = HungarianLoss(num_classes=args.num_classes, matcher=matcher, eos_coef=args.eos_coef, **factory_kwargs)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_up=20, T_mult=1., lr_delta=0.5)
    scaler = GradScaler()

    # checkpoint & tensorboard
    state = {
        'model': model,
        'matcher': matcher,
        'criterion': criterion,
        'optimizer': optimizer,
    }
    checkpoint = CKPT(args.checkpoint, state)
    writer = SummaryWriter(args.tensorboard)

    # training loop
    for epoch in range(args.epochs):
        train(trainloader, model, object_queries, pos, query_pos, optimizer, scaler, criterion, epoch, writer, args)
        score = valid(validloader, model, object_queries, pos, query_pos, criterion, epoch, writer, args)
        checkpoint.step(score)
        scheduler.step()

    torch.cuda.empty_cache()
    writer.close()
    gc.collect()


def load_batch(batch, device, non_blocking: bool = False):
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=non_blocking)
    targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in target.items()} for target in targets]
    return inputs, targets


def train(dataloader: Iterable, model: Module, object_queries: Tensor, pos: Tensor, query_pos: Tensor,
          optimizer: Optimizer, scaler: Any, criterion: Module, epoch: int, writer: Any, args: Any) -> None:
    model.train()
    criterion.train()
    losses = []
    for idx, batch in enumerate(dataloader):
        inputs, targets = load_batch(batch, args.device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs, object_queries, pos, query_pos)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        print(
            f'\r'
            f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
            f'batch[{idx + 1:{len(str(len(dataloader)))}d}/{len(dataloader)}] - '
            f'loss(current batch): {loss.item():.3f} - '
            f'loss(accumulated): {np.nanmean(losses):.3f}',
            end='',
        )
        writer.add_scalar('train / training loss (batch)', loss.item(), epoch * len(dataloader) + idx)
    print(
        f'\r'
        f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
        f'average loss: {np.nanmean(losses):.3f}'
    )
    writer.add_scalar('train / training loss (epoch)', np.nanmean(losses), epoch)
    torch.cuda.empty_cache()
    gc.collect()


def valid(dataloader: Iterable, model: Module, object_queries: Tensor, pos: Tensor, query_pos: Tensor,
          criterion: Module, epoch: int, writer:Any, args: Any) -> List[Any]:
    model.eval()
    criterion.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            inputs, targets = load_batch(batch, args.device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs, object_queries, pos, query_pos)
                loss = criterion(outputs, targets)
            losses.append(loss.item())
            print(
                f'\r'
                f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
                f'batch[{idx + 1:{len(str(len(dataloader)))}d}/{len(dataloader)}] - '
                f'loss(current batch): {loss.item():.3f} - '
                f'loss(accumulated): {np.nanmean(losses):.3f}',
                end='',
            )
            writer.add_scalar('valid / validation loss (batch)', loss.item(), epoch * len(dataloader) + idx)
    print(
        f'\r'
        f'Epoch[{epoch + 1:{len(str(args.epochs))}d}/{args.epochs}] - '
        f'average loss: {np.nanmean(losses):.3f}'
    )
    writer.add_scalar('valid / validation loss (epoch)', np.nanmean(losses), epoch)
    torch.cuda.empty_cache()
    gc.collect()
    return losses


if __name__ == '__main__':
    config = get_parser().parse_args()
    main(config)
