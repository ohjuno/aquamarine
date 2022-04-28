from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

import os.path
import torch

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader

from aquamarine.datasets.coco.functional import coco_collate_fn


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


class COCODataLoader(DataLoader):

    def __init__(self, dataset, batch_size: int = 1,
                 shuffle: bool = False, sampler=None,
                 batch_sampler=None,
                 num_workers: int = 0, collate_fn=coco_collate_fn,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super(COCODataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
