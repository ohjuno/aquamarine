from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

import os.path
import torch

from torchvision.datasets import VisionDataset

torch.multiprocessing.set_sharing_strategy('file_system')


class COCODetection(VisionDataset):

    def __init__(self, root: str, annFile: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
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
            'image_id': torch.tensor([id]),
            'labels': labels[cond],
            'bboxes': bboxes[cond],
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
