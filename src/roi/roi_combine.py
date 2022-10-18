import mindspore
from mindspore import nn
from .box_head.head import ROIBoxHead
from .mask_head.head import ROIMaskHead
from src.model_utils.matcher import Matcher

class CombinedROIHeads(nn.Cell):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, config):
        super(CombinedROIHeads, self).__init__()
        matcher = Matcher(config.roi.box_head.fg_iou,
                          config.roi.box_head.bg_iou)

        self.config = config
        self.box = ROIBoxHead(config)
        self.mask = ROIMaskHead(config, matcher, 
                                (config.roi.mask_head.resolution_w, config.roi.mask_head.resolution_h))

    def construct(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        mask_features = features

        x, detections, loss_mask = self.mask(mask_features, detections, targets)
        if loss_mask is not None:
            losses.update(loss_mask)
        return x, detections, losses
