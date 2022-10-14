from mindspore import nn, Parameter
from mindspore import numpy as np
from mindspore import dtype as mstype
from mindspore.ops import operations as P


class FixedBatchNorm2d(nn.Cell):
    def __init__(self, channels):
        super(FixedBatchNorm2d, self).__init__()
        self.weight = Parameter(np.ones(channels),"weight")
        self.bias = Parameter(np.zeros(channels),"bias")
        self.moving_mean = Parameter(np.zeros(channels),"moving_mean")
        self.moving_var = Parameter(np.ones(channels),"moving_var")

        self.cast = P.Cast()
        self.rsqrt = P.Rsqrt()

    def construct(self, inputs):
        if inputs.dtype == mstype.float16:
            self.weight = self.cast(self.weight, mstype.float16)
            self.bias = self.cast(self.bias, mstype.float16)
            self.moving_mean = self.cast(self.moving_mean, mstype.float16)
            self.moving_var = self.cast(self.moving_var, mstype.float16)
        factor = self.rsqrt(self.moving_var) * self.weight
        bias = self.bias - self.moving_mean * factor
        factor = factor.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return inputs * factor + bias
