import mindspore
from mindspore import nn, ops, Tensor

from feature_extractor import Fpn2Mlp
from inference import PostHandler
from predictor import FpnPredict
from loss import FastRCNNLoss


class ROIBoxHead(nn.Cell):
    """
    Generic Box Head class.
    """

    def __init__(self, config):
        super(ROIBoxHead, self).__init__()
        self.config = config
        self.feature_extractor = Fpn2Mlp(config)
        self.predictor = FpnPredict(config)
        self.post_processor = PostHandler(config)
        self.loss_evaluator = FastRCNNLoss(config)

    def construct(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
    
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )
