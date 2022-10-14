import mindspore
from mindspore import nn, ops, Tensor
from mindspore.ops import operations as P
from mindspore import numpy as np

class BalencedSampler(nn.Cell):
    def __init__(self, batch_size, positive_fraction):
        super(BalencedSampler, self).__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction

    def construct(self, idxs):
        positive_idx = []
        negative_idx = []

        for idxs_per_image in idxs:
            positive = ops.nonzero(idxs_per_image >= 1).squeeze(1)
            negative = ops.nonzero(idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.size(), num_pos)
            num_neg = self.batch_size - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.size(), num_neg)

            # randomly select positive and negative examples
            perm1 = P.Randperm(positive.size())(Tensor([positive.size()], mindspore.int32))[:num_pos]
            perm2 = P.Randperm(negative.size())(Tensor([negative.size()], mindspore.int32))[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = np.zeros_like(idxs_per_image, dtype=mindspore.bool_)
            neg_idx_per_image_mask = np.zeros_like(idxs_per_image, dtype=mindspore.bool_)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            positive_idx.append(pos_idx_per_image_mask)
            negative_idx.append(neg_idx_per_image_mask)

        return positive_idx, negative_idx
