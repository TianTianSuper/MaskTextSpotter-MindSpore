from mindspore import numpy as np
from mindspore import ops, nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal


class FpnPredict(nn.Cell):
    def __init__(self, config):
        super(FpnPredict, self).__init__()
        self.config = config
        class_count = config.roi.box_head.class_count
        representation_size = config.roi.box_head.mlp_dim

        self.compute_layer = nn.Dense(representation_size, class_count)
        self.use_reg = True
        if self.use_reg:
            self.pred_layer = nn.Dense(representation_size, class_count * 4, weight_init=Normal(1e-3))
    
    def construct(self, inputs):
        marks = self.compute_layer(inputs)
        if self.use_reg:
            bbox_deltas = self.bbox_pred(inputs)
        else:
            bbox_deltas = None

        return marks, bbox_deltas