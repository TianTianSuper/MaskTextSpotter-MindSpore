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
from src.masktextspotter.fpn_neck import FeatPyramidNeck
from src.model_utils.config import config

class TestResNet(object):
    @pytest.mark.fpn
    def test_normal_input(self):
        fs = [[320, 192], [160, 96], [80, 48], [24, 40], [12, 20]]
        fpn_ncek = FeatPyramidNeck([256, 512, 1024, 2048],
                                    256,
                                    5,
                                    fs)
        input_data = (np.random.normal(0,0.1,(1,c,1280//(4*2**i), 768//(4*2**i))) \
                        for i, c in enumerate([256, 512, 1024, 2048]))
        output = fpn_ncek(tuple(Tensor(t) for t in input_data))


if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/fpn_neck.html', '-m=fpn'])