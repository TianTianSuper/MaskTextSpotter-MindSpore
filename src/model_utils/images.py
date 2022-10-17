import mindspore.numpy as np
from mindspore.common.tensor import Tensor


class Images(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def get_sizes(self):
        return self.image_sizes


def to_image_list(tensors, size_divisible=0):
    if isinstance(tensors, Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, Images):
        return tensors
    elif isinstance(tensors, Tensor):
        # single tensor shape can be inferred
        assert tensors.ndim == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return Images(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = np.zeros(batch_shape)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img

        image_sizes = [im.shape[-2:] for im in tensors]

        return Images(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


# def to_image_target_list(tensors, size_divisible=0, targets=None):
#     if isinstance(tensors, Tensor) and size_divisible > 0:
#         tensors = [tensors]

#     if isinstance(tensors, Images):
#         return tensors
#     elif isinstance(tensors, Tensor):
#         # single tensor shape can be inferred
#         assert tensors.ndim == 4
#         image_sizes = [tensor.shape[-2:] for tensor in tensors]
#         return Images(tensors, image_sizes)
#     elif isinstance(tensors, (tuple, list)):
#         max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
#         if size_divisible > 0:
#             import math

#             stride = size_divisible
#             max_size = list(max_size)
#             max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
#             max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
#             max_size = tuple(max_size)

#         batch_shape = (len(tensors),) + max_size
#         batched_imgs = np.zeros(batch_shape)
#         if targets is None:
#             for img, pad_img in zip(tensors, batched_imgs):
#                 pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
#         else:
#             for img, pad_img, target in zip(tensors, batched_imgs, targets):
#                 pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
#                 if target is not None:
#                     target.set_size((pad_img.shape[2], pad_img.shape[1]))

#         image_sizes = [im.shape[-2:] for im in tensors]

#         return Images(batched_imgs, image_sizes), targets
#     else:
#         raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))

if __name__ == '__main__':
    to_image_list((Tensor((((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)))),
                   Tensor((((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)))),
                   Tensor((((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7)),
                                ((1,2,3,4,6,7),(1,2,3,4,6,7),(1,2,3,4,6,7))))))