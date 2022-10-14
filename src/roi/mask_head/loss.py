import mindspore
from mindspore import nn, ops, Tensor
from mindspore.ops import operations as P

class CharMaskRCNNLossComputation(nn.Cell):
    def __init__(self, use_weighted_loss=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.use_weighted_loss = use_weighted_loss
        self.concat = P.Concat()


    def __call__(
        self,
        proposals,
        mask_logits,
        char_mask_logits,
        mask_targets,
        char_mask_targets,
        char_mask_weights,
    ):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        mask_targets = self.concat(mask_targets)
        char_mask_targets = self.concat(char_mask_targets)
        char_mask_weights = self.concat(char_mask_weights)
        char_mask_weights = char_mask_weights.mean(dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.size() == 0 or char_mask_targets.size() == 0:
            return mask_logits.sum() * 0, char_mask_targets.sum() * 0

        mask_loss = ops.binary_cross_entropy_with_logits(
            mask_logits.squeeze(dim=1), mask_targets
        )
        if self.use_weighted_loss:
            char_mask_loss = ops.cross_entropy(
                char_mask_logits, char_mask_targets, char_mask_weights, ignore_index=-1
            )
        else:
            char_mask_loss = ops.cross_entropy(
                char_mask_logits, char_mask_targets, ignore_index=-1
            )
        return mask_loss, char_mask_loss
