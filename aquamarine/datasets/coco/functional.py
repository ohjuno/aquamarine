import torch


def align_spatial_shape_to_square(tensor_shapes):
    shape = tensor_shapes[0]
    long = max(shape)
    shape = [shape[0], long, long]
    return shape


def align_spatial_shape_with_paddings(tensors):
    tensor_shape = align_spatial_shape_to_square([list(tensor.shape) for tensor in tensors])
    batch_shape = [len(tensors)] + tensor_shape
    padded_tensors = torch.zeros(batch_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    for tensor, padded_tensor in zip(tensors, padded_tensors):
        padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
    return padded_tensors


def coco_collate_fn(batch):
    inputs, targets = list(zip(*batch))
    inputs = align_spatial_shape_with_paddings(inputs)
    return inputs, targets
