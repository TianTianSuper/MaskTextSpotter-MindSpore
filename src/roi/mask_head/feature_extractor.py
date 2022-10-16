import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import HeNormal
from ..pooler import Pooler

class MaskRCNNFPNFeatureExtractor(nn.Cell):
    """
    Heads for FPN for classification
    """

    def __init__(self, config):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        # resolution = config.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        resolution_h = config.roi.mask_head.resolution_h
        resolution_w = config.roi.mask_head.resolution_w

        scales = config.roi.mask_head.scales
        sampling_rate = config.roi.sample_rate
        pooler = Pooler(
            output_size=(resolution_h, resolution_w),
            scales=scales,
            sampling_ratio=sampling_rate,
        )
        input_size = config.resnet_out_channels[-1]
        self.pooler = pooler

        layers = config.roi.mask_head.conv_layers

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            weight_init = HeNormal(mode='fan_out', nonlinearity="relu")
            module = nn.Conv2d(next_feature, layer_features, 3, stride=1, padding=1, weight_init=weight_init, bias_init='zero')
            self.insert_child_to_cell(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
