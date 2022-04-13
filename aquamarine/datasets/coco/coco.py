from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

import os.path
import torch

from torchvision.datasets import VisionDataset


class COCODetection(VisionDataset):

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(COCODetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = sorted([i for i in self.coco.imgs.keys() if len(self.coco.getAnnIds(i, iscrowd=False)) != 0])

    def __load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self.__load_image(id)
        annotations = self.__load_target(id)

        bboxes = [annotation['bbox'] for annotation in annotations]
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        bboxes[:, 2:] += bboxes[:, :2]
        bboxes[:, 0::2].clamp_(min=0, max=image.size[0])
        bboxes[:, 1::2].clamp_(min=0, max=image.size[1])
        labels = [annotation['category_id'] for annotation in annotations]
        labels = torch.tensor(labels, dtype=torch.float32)

        cond = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])
        target = {
            'image_id': id,
            'bboxes': bboxes[cond],
            'labels': labels[cond],
            'area': torch.tensor([annotation['area'] for annotation in annotations])[cond],
            'iscrowd': torch.tensor([annotation['iscrowd'] for annotation in annotations])[cond],
            'size': torch.as_tensor([int(image.size[1]), int(image.size[0])]),
            'orig_size': torch.as_tensor([int(image.size[1]), int(image.size[0])])
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = collate_inputs_from_inputs_list(batch[0])
    return tuple(batch)


def collate_inputs_from_inputs_list(tensors):
    max_size = max_by_axis([list(image.shape) for image in tensors])
    batch_shape = [len(tensors)] + max_size
    dtype = tensors[0].dtype
    device = tensors[0].device
    paddings = torch.zeros(batch_shape, dtype=dtype, device=device)
    for image, padding in zip(tensors, paddings):
        padding[: image.shape[0], : image.shape[1], : image.shape[2]].copy_(image)
    return paddings


def max_by_axis(l):
    maxes = l[0]
    for sublist in l[1:]:
        for idx, item in enumerate(sublist):
            maxes[idx] = max(maxes[idx], item)
    return maxes


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms

    coco_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    coco_trainset = COCODetection(
        root='/mnt/datasets/coco/train2017',
        annFile='/mnt/datasets/coco/annotations/instances_train2017.json',
        transform=coco_transform,
    )
    coco_trainloader = DataLoader(coco_trainset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

    for idx, batch in enumerate(coco_trainloader):
        inputs, targets = batch
        breakpoint()
