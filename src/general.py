import mindspore
from mindspore import nn

from src.masktextspotter.resnet50 import ResNetFea
from src.masktextspotter.spn import SEG
from src.roi.roi_combine import CombinedROIHeads

from .model_utils.images import to_image_list

class GeneralLoss(nn.Cell):
    def construct(self, losses_dict):
        losses = [v for v in losses_dict.values()]
        total_loss = sum(losses)
        return total_loss

class MaskTextSpotter3(nn.Cell):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, config):
        super(MaskTextSpotter3, self).__init__()
        self.config = config
        self.backbone = ResNetFea(config)
        self.proposal = SEG(config)
        self.roi_heads = CombinedROIHeads(config)

    def construct(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        (proposals, proposal_losses), fuse_feature = self.proposal(images, features, targets)
        x, result, detector_losses = self.roi_heads(fuse_feature, proposals, targets)

        if self.training:
            losses = {}
            if self.roi_heads is not None:
                losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
