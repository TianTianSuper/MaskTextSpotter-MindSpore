import sys
sys.path.append('/home/tiantian/Documents/internship/MindSpore/OCR/MaskTextSpotter-MindSpore/')

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from src.masktextspotter.resnet50 import ResNetFea, ResidualBlockUsing, _BatchNorm2dInit
from src.model_utils.norm import FixedBatchNorm2d

class TestResNet(object):
    def test_main(self):
        resnet = ResNetFea(ResidualBlockUsing,
                                  [3, 4, 6, 3],
                                  [64, 256, 512, 1024],
                                  [256, 512, 1024, 2048],
                                  False)
        input = Tensor(np.ones((64,3,7,7)))
        output = resnet(input)[-1].asnumpy()

if __name__ == '__main__':
    pytest.main(['-vs','--html=unittest/results/report.html'])