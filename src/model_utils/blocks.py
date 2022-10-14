from mindspore import nn

class ConvBnReluBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, has_bias=False, weight_init='HeUniform', bias_init='zeros'):
        super(ConvBnReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, pad_mode='pad',
                              has_bias=has_bias, weight_init='HeUniform', bias_init='zeros')
        self.bn = nn.BatchNorm2d(out_channels) # Q2: 无法初始化权重和偏置
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
