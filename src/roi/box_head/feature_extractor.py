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
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        if self.config.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
            input_size = (config.MODEL.BACKBONE.OUT_CHANNELS + 1) * resolution ** 2
        else:
            input_size = config.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
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
                scaled_mask = cropped_mask.resize((self.config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, self.config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION))
                mask = scaled_mask.convert(mode="mask")
                masks.append(mask)
        if len(masks) == 0:
            if self.config.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
                inputs = self.s_concat((inputs, np.ones((inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]))))
            return inputs
        masks = ops.stack(masks, dim=0).to(x.device, dtype=mstype.float32)
        if self.config.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
            inputs = self.s_concat((inputs, masks.expend_dims(1)))
            return inputs
        if self.config.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'ATTENTION':
            # x_cat = cat([x, masks.expend_dims(1)], dim=1)
            # attention = self.attention(x_cat)
            # x = x * attention
            return inputs
        soft_num = self.config.MODEL.ROI_BOX_HEAD.SOFT_MASKED_FEATURE_RATIO
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
        if self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE:
            x = self.feature_mask(x, proposals)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x
