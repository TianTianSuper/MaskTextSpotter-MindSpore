'''
获取由pytorch输出的结果, 固定相同的输入, 测试pytorch与mindspore的输出结果是否一致
'''

import pytest

import numpy as np
import pickle as pkl
from mindspore.common.initializer import Normal
from mindspore import nn, Tensor
from mindspore import dtype as mstype

def test_conv2d():
    np.random.seed(2)
    weight_init = Normal(sigma=1)
    conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, has_bias=False, weight_init=weight_init, pad_mode='same')
    output = conv(Tensor(np.random.random(65536).reshape(1,256,16,16), dtype=mstype.float32)).asnumpy()
    with open('./conv2d_case.pkl', 'rb') as f:
        output_standard = pkl.load(f)
    assert (output == output_standard).all()

if __name__ == '__main__':
    pytest.main(['-vs'])