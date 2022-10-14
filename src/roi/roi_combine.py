import mindspore
from mindspore import nn

class CombinedROIHeads(nn.Cell):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, config, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.config = config.copy()
        if config.MODEL.MASK_ON and config.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def construct(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.config.MODEL.MASK_ON or self.config.SEQUENCE.SEQ_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads,
            # then we can reuse the features already computed
            if (
                self.training
                and self.config.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return
            # the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            if loss_mask is not None:
                losses.update(loss_mask)
        return x, detections, losses
