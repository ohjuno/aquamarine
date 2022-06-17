import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import aquamarine.datasets.coco.transforms as transforms

from torch.utils.data import DataLoader
from aquamarine.datasets import COCODetection, coco_collate_fn
from aquamarine.engine import update, run
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher


def get_parser():
    parser = argparse.ArgumentParser('DETR Trainer', add_help=False)

    # dataset and dataloader
    parser.add_argument('--train_root', default='/mnt/datasets/coco/train2017', type=str)
    parser.add_argument('--train_annFile', default='/mnt/datasets/coco/annotations/instances_train2017.json', type=str)
    parser.add_argument('--valid_root', default='/mnt/datasets/coco/val2017', type=str)
    parser.add_argument('--valid_annFile', default='/mnt/datasets/coco/annotations/instances_val2017.json', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_worker', default=8, type=int)
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
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--tensorboard', default=None, type=str)

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
    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=args.num_worker, pin_memory=True, collate_fn=coco_collate_fn)
    validloader = DataLoader(validset, args.batch_size, False, num_workers=args.num_worker, pin_memory=True, collate_fn=coco_collate_fn)

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

    # positional encodings
    pos = nn.Parameter(torch.empty((1, int((args.resolution / 32) ** 2), 512), **factory_kwargs))
    query_pos = nn.Parameter(torch.empty((1, args.num_queries, 512), **factory_kwargs))

    # optimizer, criterion, scheduler
    matcher = HungarianMatcher(lamb_labels=args.lamb_labels, lamb_bboxes=args.lamb_bboxes, lamb_geniou=args.lamb_geniou)

    valid_process_fn = run(model, load_batch, device=args.device)

    # training loop
    for epoch in range(args.epochs):
        valid(validloader, valid_process_fn)


def load_batch(batch, device, non_blocking: bool = False):
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=non_blocking)
    targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in target.items()} for target in targets]
    return inputs, targets


def valid(dataloader, process_fn):
    for idx, batch in enumerate(dataloader):
        inputs, targets, outputs = process_fn(batch)
        breakpoint()


if __name__ == '__main__':
    main(get_parser().parse_args())
