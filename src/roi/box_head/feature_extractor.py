from mindspore import numpy as np
from mindspore import ops, nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype

from ..pooler import Pooler
from ...model_utils.tools import SafeConcat

class Fpn2Mlp(nn.Cell):
    def __init__(self, config):
        super(Fpn2Mlp, self).__init__()
        self.config = config
        self.resolution = config.roi.box_head.resolution
        scales = config.roi.box_head.scales
        sampling_rate = config.roi.box_head.sample_rate
        pooler = Pooler(
            output_size=(self.resolution, self.resolution),
            scales=scales,
            sampling_ratio=sampling_rate,
        )

        input_size = config.resnet_out_channels[-1] * self.resolution ** 2
        representation_size = config.roi.box_head.mlp_dim
        self.pooler = pooler
        self.fc6 = nn.Dense(input_size, representation_size, weight_init='HeUniform', bias_init='zero')
        self.fc7 = nn.Dense(representation_size, representation_size, weight_init='HeUniform', bias_init='zero')

        # utils
        self.s_concat = SafeConcat(1)
        self.relu = P.ReLU()
    
    def feature_mask(self, inputs, proposals):
        masks = []
        for proposal in proposals:
            segmentation_masks = proposal.get_field("masks")
            boxes = proposal.bbox
            for segmentation_mask, box in zip(segmentation_masks, boxes):
                cropped_mask = segmentation_mask.crop(box)
                scaled_mask = cropped_mask.resize((self.resolution, self.resolution))
                mask = scaled_mask.convert(mode="mask")
                masks.append(mask)
        if len(masks) == 0:
            return inputs
        masks = ops.stack(masks, dim=0).to(x.device, dtype=mstype.float32)
        soft_num = self.config.roi.box_head.soft_mask_feat_rate
        if soft_num > 0:
            if soft_num < 1.0:
                x = x * (soft_num + (1 - soft_num) * masks.expend_dims(1))
            else:
                x = x * (1.0 + soft_num * masks.expend_dims(1))
        else:
            x = x * masks.expend_dims(1)
        return x
    
    def construct(self, inputs, proposals):
        x = self.pooler(inputs, proposals)
        x = self.feature_mask(x, proposals)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x
