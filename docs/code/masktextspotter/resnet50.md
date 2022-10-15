# ResNet50 说明文档

## 概述

ResNet作为深度学习的基础模型，在本项目中作为backbone的一部分，用来提取图像的特征。


## 比较与分析

官方源码提供的Resnet实现比较灵活，支持设定多种模式改变resnet的结构获取想要输出的值。但是这种灵活性在其本身实现模型的时候根本没有运用，因为调用的只是resnet50的结构，且不存在需要获取中间层输出的情况。因此在MindSpore的实现中，直接实现resnet50即可，无需对resnet的结构做过多的灵活处理。

而由于ResNet在多个深度学习都有实现，在MindSpore中亦是如此。再者，本项目构建的模型与MaskRCNN的实现已有一些共通之处，比如backbone的结构是相差无几的，都包含了resnet。因此直接采用MindSpore Modelzoo中的[MaskRCNN](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/src/maskrcnn/resnet50.py)中实现的Resnet50作为本项目实现的backbone。

## 代码说明

### 模块作用

`def weight_init_ones(shape)`: 定义一个权重初始化的方法
`def _conv(...)`: 返回初始化权重后的卷积层
`def _BatchNorm2dInit(...)`: 对应pytorch实现中的FrozenBatchNorm2d，是构成resnet重要的一部分

`class ResNetFea(nn.Cell)`: resnet的主要结构实现
`class ResidualBlockUsing(nn.Cell)`: 残差块结构的主要实现

### 与pytorch的区别

最大的区别在于先前提到的灵活性。因为pytorch实现的resnet要求在同一个文件下可以返回不同结构、深度和版本的resnet，因此加入了大量预定义的参数以此来定义模型。但是本项目只要求实现resnet50即可。除了实现首层所谓的stem(pytorch中的叫法)，其余四层可以固定下来。

#### PyTorch __init__
```python
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        ## 引入大量预定义参数
        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC] 
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY] 
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC] # "BottleneckWithFixedBatchNorm"

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS # 1
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP # 64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS # 64
        stage2_bottleneck_channels = num_groups * width_per_group # 64
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS # 256
        self.stages = []
        self.return_features = {}
        ## 灵活构建，但是本项目不需要
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1] # (False, False, False, False)
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN, # False
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS, # 1
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT) # 2

```

#### MindSpore __init__
```python
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 weights_update=False):
        super(ResNetFea, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")
        # 此处即所谓的stem
        bn_training = False
        self.conv1 = _conv(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = _BatchNorm2dInit(64, affine=bn_training, use_batch_statistics=bn_training)
        self.relu = P.ReLU()
        self.maxpool = P.MaxPool(kernel_size=3, strides=2, pad_mode="SAME")
        self.weights_update = weights_update

        if not self.weights_update:
            self.conv1.weight.requires_grad = False
        # 固定四层layer
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=1,
                                       training=bn_training,
                                       weights_update=self.weights_update)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=2,
                                       training=bn_training,
                                       weights_update=True)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=2,
                                       training=bn_training,
                                       weights_update=True)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=2,
                                       training=bn_training,
                                       weights_update=True)
```

## 心得

虽然ResNet比较通用和出名，但是针对具体的需求，实现方式都会不一样。但是选择最适合项目本身的实现方法即可。其实pytorch的写法有点复杂化了，不过因为那是maskrcnn标准实现，可以直接调用的代码，必须保证通用性。这也无可厚非。