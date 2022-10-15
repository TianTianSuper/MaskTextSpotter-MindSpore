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
    def test_normal_input(self):
        resnet = ResNetFea(ResidualBlockUsing,
                                  [3, 4, 6, 3],
                                  [64, 256, 512, 1024],
                                  [256, 512, 1024, 2048],
                                  False)
        input = Tensor(np.ones((64,3,7,7)))
        output = resnet(input)[-1].asnumpy()

    def test_not_2d(self):
        with pytest.raises(ValueError):
            resnet = ResNetFea(ResidualBlockUsing,
                                    [3, 4, 6, 3],
                                    [64, 256, 512, 1024, 2048],
                                    [256, 512, 1024, 2048, 4096],
                                    False)
            input = Tensor(np.ones((64,3,7,7)))
            output = resnet(input)
    
    def test_wrong_input_shape(self):
        with pytest.raises(ValueError):
            resnet = ResNetFea(ResidualBlockUsing,
                                    [3, 4, 6, 3],
                                    [64, 256, 512, 1024],
                                    [256, 512, 1024, 2048],
                                    False)
            input = Tensor(np.ones((3,7,7)))
            output = resnet(input)

    def test_wrong_Cin(self):
        with pytest.raises(RuntimeError):
            resnet = ResNetFea(ResidualBlockUsing,
                                    [3, 4, 6, 3],
                                    [64, 256, 512, 1024],
                                    [256, 512, 1024, 2048],
                                    False)
            input = Tensor(np.ones((64,4,7,7)))
            output = resnet(input)

    def test_other_wh(self):
        resnet = ResNetFea(ResidualBlockUsing,
                                [3, 4, 6, 3],
                                [64, 256, 512, 1024],
                                [256, 512, 1024, 2048],
                                False)
        input = Tensor(np.ones((64,3,128,128)))
        output = resnet(input)

if __name__ == '__main__':
    pytest.main(['-vs','--html=unittest/results/resnet50.html'])