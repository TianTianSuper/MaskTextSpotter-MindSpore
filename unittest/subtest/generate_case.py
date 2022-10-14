'''
在固定输入的前提下, 使用torch输出对应的结果
'''

import numpy as np
import torch
from torch import nn
import pickle as pkl

np.random.seed(2)
layer = torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
nn.init.normal_(layer.weight)
input_1 = torch.Tensor((np.random.random(65536).reshape(1,256,16,16))).to(torch.float32)
output = layer(input_1)
with open('./conv2d_case.pkl', 'wb') as f:
    pkl.dump(output.detach().numpy(), f)