import torch
import torch.nn as nn
import torchvision.models as models

from aquamarine.models import DETR, DETRTransformer


if __name__ == '__main__':

    backbone = nn.Sequential(*list(models.resnet50().children()))[:-2]
    detector = DETRTransformer(512, 8, 2048, 6, 6)
    model = DETR(backbone, detector, num_classes=100, num_queries=100)
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 640, 640, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(
        model,
        x,
        'detr.onnx',
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
