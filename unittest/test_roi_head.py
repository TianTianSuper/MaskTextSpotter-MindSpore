import os
import sys
import mindspore
sys.path.append(os.getcwd())

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import normal
from mindspore.common.initializer import Normal
from src.roi.box_head.feature_extractor import Fpn2Mlp
from src.roi.box_head.head import ROIBoxHead
from src.roi.box_head.inference import PostHandler
from src.roi.box_head.loss import FastRCNNLoss
from src.roi.box_head.predictor import FpnPredict
from src.model_utils.config import config

class TestRoiBox(object):
    @pytest.mark.rbh
    def test_fpn2mlp(self):
        # Warning: No enough resource to test
        with open('unittest/case/target.pkl', 'rb') as f:
            target = pkl.load(f)
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        part = Fpn2Mlp(config)
        out = part(Tensor(img), target)


if __name__ == '__main__':
    from src.model_utils.config import config
    from mindspore import context
    context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)

    pytest.main(['-vv','-s','--html=unittest/results/roi_box_head.html', '-m=rbh'])