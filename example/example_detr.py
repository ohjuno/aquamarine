import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.models import DETR, DETRTransformer, HungarianMatcher
from aquamarine.modules import HungarianLoss


if __name__ == '__main__':
    # configurations
    root = '/mnt/datasets/coco/train2017'
    annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 4

    # dataloader
    trainset = COCODetection(root=root, annFile=annFile, transform=transform)
    trainloader = COCODataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # model
    backbone = nn.Sequential(*list(models.resnet18().children()))[:-2]
    detector = DETRTransformer(512, 8, 2048, 6, 6)
    model = DETR(backbone, detector, num_classes=100, num_queries=100)

    # criterion
    matcher = HungarianMatcher(lamb_labels=1., lamb_bboxes=5., lamb_geniou=2.)
    criterion = HungarianLoss(num_classes=100, matcher=matcher, eos_coef=0.1)

    # data
    inputs, targets = next(iter(trainloader))
    outputs = model(inputs)

    # debug
    losses = criterion(outputs, targets)
    breakpoint()
