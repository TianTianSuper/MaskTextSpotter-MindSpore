import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


class SEGLoss(nn.Cell):
    def __init__(self, config):
        super(SEGLoss, self).__init__()
        self.eps = 1e-6
        self.config = config

        # utils
        self.reduce_sum = P.ReduceSum()
    
    def dice_loss(self, pred, gt, m):
        intersection = self.reduce_sum(pred * gt * m)
        union = self.reduce_sum(pred * m) + self.reduce_sum(gt * m) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss

    def project_masks_on_image(self, mask_polygons, labels, shrink_ratio, image_size):
        seg_map, training_mask = mask_polygons.convert_seg_map(
            labels, shrink_ratio, image_size, True
        )
        return Tensor(seg_map), Tensor(training_mask)

    def prepare_targets(self, targets, image_size):
        segms = []
        training_masks = []
        for target_per_image in targets:
            segmentation_masks = target_per_image.get_field("masks")
            labels = target_per_image.get_field("labels")
            seg_maps_per_image, training_masks_per_image = self.project_masks_on_image(
                segmentation_masks, labels, self.config.sequence.shrink_rate, image_size
            )
            segms.append(seg_maps_per_image)
            training_masks.append(training_masks_per_image)
        return ops.stack(segms), ops.stack(training_masks)

    def construct(self, preds, targets):
        image_size = (preds.shape[1], preds.shape[2])
        segm_targets, masks = self.prepare_targets(targets, image_size)
        segm_targets = segm_targets.astype(mindspore.float32).squeeze()
        masks = masks.astype(mindspore.float32)
        seg_loss = self.dice_loss(preds, segm_targets, masks)
        return seg_loss
