from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.datasets.coco.transforms import Compose, Normalize, Resize, ToTensor
from aquamarine.datasets.utils import visualize_bounding_boxes_on_batch


def main():
    # configurations
    root = '/mnt/datasets/coco/train2017'
    annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
    resolution = 640
    coco_transforms = Compose([
        Resize(resolution),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # `Normalize` converts the format of bounding box from `xyxy` to `cxcywh`
    ])

    # init dataset and dataloader
    trainset = COCODetection(root=root, annFile=annFile, transforms=coco_transforms)
    trainloader = COCODataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)  # num_worker=0 for debugging

    # get single batch
    inputs, targets = next(iter(trainloader))
    visualize_bounding_boxes_on_batch((inputs, targets))


if __name__ == '__main__':
    main()
