import torch
import torch.nn as nn
import torchvision.models as models
import aquamarine.datasets.coco.transforms as transforms

from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.datasets.utils import visualize_bounding_boxes_on_batch
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher
from aquamarine.modules import HungarianLoss


if __name__ == '__main__':
    # configurations
    root = '/mnt/datasets/coco/train2017'
    annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
    batch_size = 16
    resolution = 640
    num_classes = 100
    num_queries = 100

    # dataloader
    coco_transforms = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # `Normalize` converts the format of bounding box from `xyxy` to `cxcywh`
    ])
    trainset = COCODetection(root=root, annFile=annFile, transforms=coco_transforms)
    trainloader = COCODataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # model
    backbone = nn.Sequential(*list(models.resnet50().children()))[:-2]
    detector = DETRTransformer(512, 8, 2048, 6, 6)
    model = DETR(backbone, detector, num_classes=num_classes, num_queries=num_queries)

    # object queries & positional encodings
    object_queries = torch.zeros(batch_size, num_queries, 512)
    pos = nn.Parameter(torch.empty((batch_size, int((resolution / 32) ** 2), 512)))
    query_pos = nn.Parameter(torch.empty((batch_size, num_queries, 512)))

    # criterion
    matcher = HungarianMatcher(lamb_labels=1., lamb_bboxes=5., lamb_geniou=2.)
    criterion = HungarianLoss(num_classes=100, matcher=matcher, eos_coef=0.1)

    # data
    # target bounding box format: `cxcywh`
    inputs, targets = next(iter(trainloader))
    visualize_bounding_boxes_on_batch((inputs, targets))
    outputs = model(inputs, object_queries, pos, query_pos)
    outputs = torch.nan_to_num(outputs, nan=1e-5)
    outputs = torch.split(outputs, [num_classes + 1, 4], dim=-1)
    outputs = dict(labels=outputs[0], bboxes=outputs[1])

    # debug
    losses = criterion(outputs, targets)
    breakpoint()
