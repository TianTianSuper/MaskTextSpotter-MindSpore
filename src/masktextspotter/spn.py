import os
import sys
sys.path.append(os.getcwd())

from mindspore import nn, Parameter
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import context

from src.model_utils.blocks import ConvBnReluBlock
from src.masktextspotter.inference import SEGPostHandler
from src.masktextspotter.loss import SEGLoss


if context.get_context("device_target") == "Ascend":
    ms_cast_type = mstype.float16
else:
    ms_cast_type = mstype.float32


class FpnBlock(nn.Cell):
    def __init__(self, in_channels, conv_out_channels, kernel_size=3, stride=1, padding=1, has_bias=False, 
                 scale=1):
        super(FpnBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, pad_mode='pad',
                              has_bias=has_bias, weight_init='HeUniform')
        self.sample = nn.ResizeBilinear() # Q1: 无法使用最临近方法做上采样，已用这个代替
        self.scale = scale
        # self.shape = Parameter(Tensor(), name="shape", requires_grad=False)

    def construct(self, x):
        x = self.conv(x)
        # if self.shape == None:
        #     front = x.shape[0:2]
        #     later = tuple([i*self.scale for i in x.shape[2:]]) # TODO 3: 待优化的处理，实现只输入scale_factor就可以初始化，而且在init里面初始化好了
        #     self.shape = front + later            
        x = self.sample(x, scale_factor=self.scale)
        return x

class SEGHead(nn.Cell):
    def __init__(self, in_channels, config):
        super(SEGHead, self).__init__()
        self.config = config
        n_dims = 256
        self.fpn_block_5 = FpnBlock(n_dims, 64, scale=8)
        self.fpn_block_4 = FpnBlock(n_dims, 64, scale=4)
        self.fpn_block_3 = FpnBlock(n_dims, 64, scale=2)
        self.fpn_block_2 = nn.Conv2d(n_dims, 64, 3, pad_mode='pad', padding=1, weight_init='HeUniform')
        self.seg_block = nn.SequentialCell([
            ConvBnReluBlock(in_channels, 64),
            nn.Conv2dTranspose(64, 64, 2, 2),
            nn.BatchNorm2d(64), # Q2: 因为这个权重没办法初始化，因此先写在一起，构建两种写法
            nn.ReLU(), # C1: 作者代码加了个True
            nn.Conv2dTranspose(64, 1, 2, 2),
            nn.Sigmoid()
        ])
        # C2: 此后对应源代码的PPM，还有权重初始化，个人认为mindspore不用这么做
        # 因为源代码预设参数的值设置成了false，处于禁止状态

        self.concat = P.Concat(1)
    
    def construct(self, x):
        # C3: 此处对应源代码PPM的识别处理，暂未明白什么是PPM
        f = x[-2]
        p5 = self.fpn_block_5(f)
        p4 = self.fpn_block_4(x[-3])
        p3 = self.fpn_block_3(x[-4])
        p2 = self.fpn_block_2(x[-5])
        fuse = self.concat((p5, p4, p3, p2))
        out = self.seg_block(fuse)
        return out, fuse



class SEG(nn.Cell):
    def __init__(self, config, train_status=True):
        super(SEG, self).__init__()

        in_channels = config.fpn_out_channels
        self.head = SEGHead(in_channels, config)
        self.box_selector = SEGPostHandler(config, train_status=train_status)
        self.loss = SEGLoss(config)
        self.train_status = train_status

    def construct(self, images, features, targets=None):
        preds, fuse_feature = self.head(features)
        # anchors = self.anchor_generator(images, features)
        image_shapes = images.get_sizes()
        if self.train_status:
            boxes = self.box_selector(preds, image_shapes, targets)
            losses = self.loss(preds, targets)
            losses = {"loss_seg": losses}
            return boxes, losses
        else:
            boxes, rotated_boxes, polygons, scores = self.box_selector(preds, image_shapes)
            results = {'rotated_boxes': rotated_boxes, 'polygons': polygons, 'preds': preds, 'scores': scores}
            return boxes, results

if __name__ == '__main__':
    from src.model_utils.config import config
    import numpy as np
    from mindspore import context
    context.set_context(device_target='GPU',mode=context.PYNATIVE_MODE)


    test_part = SEGHead(256, config)
    input_data = Tensor(np.ones(327680).reshape((1, 5, 256, 16, 16)).T, mstype.float32)
    output = test_part(input_data)