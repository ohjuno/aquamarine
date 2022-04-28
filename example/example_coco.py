import torchvision.transforms as transforms

from aquamarine.datasets.coco import COCODetection, COCODataLoader
from aquamarine.datasets.utils import visualize_bounding_boxes_on_batch

# configurations
root = '/mnt/datasets/coco/train2017'
annFile = '/mnt/datasets/coco/annotations/instances_train2017.json'
transform = transforms.Compose([transforms.ToTensor()])


if __name__ == '__main__':
    trainset = COCODetection(root=root, annFile=annFile, transform=transform)
    trainloader = COCODataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    # get single batch
    inputs, targets = next(iter(trainloader))
    visualize_bounding_boxes_on_batch((inputs, targets))

    # make training loop
    # for idx, batch in enumerate(trainloader):
    #     pass

    breakpoint()
