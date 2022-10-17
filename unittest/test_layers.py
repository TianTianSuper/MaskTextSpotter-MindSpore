import os
import sys

sys.path.append(os.getcwd())

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import normal
from mindspore.common.initializer import Normal
from src.model_utils.norm import FixedBatchNorm2d
from src.model_utils.blocks import ConvBnReluBlock
from src.model_utils.bbox_ops import *
from src.model_utils.config import config
import pickle as pkl


class TestLayer(object):
    @pytest.mark.layer
    def test_norm(self):
        context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)
        nm = FixedBatchNorm2d(256)
        assert type(nm) == FixedBatchNorm2d
        out = nm(Tensor(np.ones(256)))
    
    @pytest.mark.layer
    def test_conv_block(self):
        context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)
        nm = ConvBnReluBlock(256, 256)
        assert type(nm) == ConvBnReluBlock
        out = nm(Tensor(np.ones((256,256,3,3)), mstype.float32))
        with pytest.raises(ValueError):
            nm(Tensor(np.ones((256,256)), mstype.float32))

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/layers.html', '-m=layer'])