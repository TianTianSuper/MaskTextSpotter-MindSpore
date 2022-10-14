import sys
sys.path.append('/root/OCR/MaskTextSpotter-MindSpore/')

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from src.masktextspotter.resnet50 import ResNetFea, ResidualBlockUsing, _BatchNorm2dInit
from src.model_utils.norm import FixedBatchNorm2d

class TestResNet(object):
    # def test_main(self):
    #     resnet = ResNetFea(ResidualBlockUsing,
    #                               [3, 4, 6, 3],
    #                               [64, 256, 512, 1024],
    #                               [256, 512, 1024, 2048],
    #                               False)
    #     input = Tensor(np.ones((64,3,7,7)))
    #     output = resnet(input)[-1].asnumpy()
    #     with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/cases/resnet_1.pkl', 'rb') as f:
    #         output_standard = pkl.load(f)
    #     with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/results/resnet.pkl', 'wb') as f:
    #         pkl.dump(output, f)
    #     assert (output == output_standard).all()
    
    # def test_norm(self):
    #     np.random.seed(2)
    #     norm = _BatchNorm2dInit(256)
    #     output = norm(Tensor(np.random.random(1024).reshape(4,256,1,1),mstype.float32)).asnumpy()
    #     with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/cases/norm.pkl', 'rb') as f:
    #         output_standard = pkl.load(f)
    #     assert (output == output_standard).all()
    
    def test_class_norm(self):
        np.random.seed(2)
        norm = FixedBatchNorm2d(256)
        output = norm(Tensor(np.random.random(1024).reshape(4,256,1,1),mstype.float32)).asnumpy()
        with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/cases/norm.pkl', 'rb') as f:
            output_standard = pkl.load(f)
        assert (output == output_standard).all()
    
    def test_conv2d(self):
        np.random.seed(2)
        weight_init = Normal(sigma=1)
        conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, has_bias=False, weight_init=weight_init, pad_mode='valid')
        output = conv(Tensor(np.random.random(65536).reshape(1,256,16,16), dtype=mstype.float32)).asnumpy()
        with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/cases/conv2d.pkl', 'rb') as f:
            output_standard = pkl.load(f)
        with open('/root/OCR/MaskTextSpotter-MindSpore/unittest/results/conv2d.pkl', 'wb') as f:
            pkl.dump(output, f)
        assert (output == output_standard).all()

if __name__ == '__main__':
    pytest.main(['-vs'])