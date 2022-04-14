from typing import List, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes

plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (50, 50)


def show(images: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]) -> None:
    if not isinstance(images, list):
        images = [images]
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for idx, image in enumerate(images):
        image = image.detach()
        image = F.to_pil_image(image)
        axs[0, idx].imshow(np.asarray(image))
        axs[0, idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def cast_float32_to_uint8(t: torch.Tensor):
    return (t * 255).type(torch.uint8)


def visualize_bounding_boxes_on_batch(batch):
    images, targets = batch
    images = cast_float32_to_uint8(images)
    results = []
    for image, target in zip(images, targets):
        results.append(draw_bounding_boxes(image, target['bboxes']))
    show(results)
