# fpn 说明文档

------

注意：此模块还需要进一步考虑输入和输出，讨论feature_shape的去留

------

## 概述

FPN 全称 Feature Pyramid Network，顺序译为特征金字塔网络。主要用于提取不同尺度的特征，并实现特征融合。

## 分析

本项目构建的模型与MaskRCNN的实现有共通之处，比如backbone的结构是相差无几的，都包含了resnet和fpn的实现。因此直接采用MindSpore Modelzoo中的[MaskRCNN](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/src/maskrcnn/fpn_neck.py)中实现的fpn_neck作为本项目实现的backbone。

## 代码说明

### 模块作用

`def bias_init_zeros(shape)`: 定义偏置初始化的方法

`def _conv(...)`: 返回初始化权重和偏置后的卷积层

`class FeatPyramidNeck(nn.Cell)`: fpn的主要结构实现

### 与pytorch的区别

#### 权重初始化的方式不同

在定义卷积块的时候，pytorch对于初始化权重采取的策略是`kaiming_uniform`，但是mindspore maskrcnn的针对于此处小模块的实现，使用`XavierUniform`做初始化。

对于pytorch在实现中引入的参数，其实可以忽略，因为在它的config文件中已经将这两个设置成False，在自定义的yaml也没有相应的开关。因此可以认为此处构造卷积层(块)的逻辑是相同的。

maskrcnn_benchmark/config/defaults.py

~~~python
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False
~~~

##### PyTorch

maskrcnn_benchmark/modeling/make_layers.py

```python
def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv
```

##### MindSpore

src/masktextspotter/fpn_neck.py

~~~python
def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = initializer(" ", shape=shape, dtype=mstype.float32)
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)

~~~

#### 类名不同

|      | MindSpore       | PyTorch |
| ---- | --------------- | ------- |
| 类名 | FeatPyramidNeck | FPN     |

#### 传入参数不同

| MindSpore      | PyTorch         |
| -------------- | --------------- |
| in_channels    | in_channel_list |
| out_channels   | out_channels    |
| num_outs       |                 |
| feature_shapes |                 |
|                | conv_block      |
|                | top_blocks      |

#### feature_shape

这一块mindspore需要从预定义的参数中获取，但是pytorch的不用

##### PyTorch

maskrcnn_benchmark/modeling/backbone/fpn.py

```python
    def forward(self, x):
        # ...
    	for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
        # ...
```

##### MindSpore

src/masktextspotter/fpn_neck.py

```python
    def __init__(...):
        # ...
    	for _, channel in enumerate(in_channels):
            l_conv = _conv(channel, out_channels, kernel_size=1, stride=1,
                           padding=0, pad_mode='valid').to_float(self.cast_type)
            fpn_conv = _conv(out_channels, out_channels, kernel_size=3, stride=1,
                             padding=0, pad_mode='same').to_float(self.cast_type)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.interpolate1 = P.ResizeBilinear(feature_shapes[2])
        self.interpolate2 = P.ResizeBilinear(feature_shapes[1])
        self.interpolate3 = P.ResizeBilinear(feature_shapes[0])
```

#### 模块后部定义MaxPool的方式不同(计划修改成PyTorch那样的)

pytorch中，要求使用`fpn_module.LastLevelMaxPool`作为`top_block`；而在mindspore中直接在模块后部加入此层。

##### pytorch

maskrcnn_benchmark/modeling/backbone/fpn.py

先做一个判断，在选择对应的层

```python
	def forward(..):
        # ...
		if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
```

两个不同的类

```python
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
```

##### MindSpore

```python
	def construct(...):
        # ...
		for i in range(self.num_outs - self.fpn_layer):
            outs = outs + (self.maxpool(outs[3]),)
        return outs
```

## 心得

Backbone与maskrcnn有诸多共同点，很多逻辑都是直接参考mindspore maskrcnn的成果，开发速度较快。与pytorch对比，其实发现舍弃了通用性之后，针对单个模型而言，mindspore的代码更加美观，较容易理解，因为少了许多需要传入的预定义参数。

## 更多

[语义分割网络 - FPN 结构及代码](https://juejin.cn/post/6844903950336917512)

[深度学习：Xavier and Kaiming Initialization](https://zhuanlan.zhihu.com/p/64464584)