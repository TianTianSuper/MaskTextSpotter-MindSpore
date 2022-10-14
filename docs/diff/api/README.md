# 算子/API 差异分析文档

目前已找到8个算子/API与PyTorch存在一定的差异，虽然会带来一些困扰和问题，但是总体可控。

## Tensor的API

| MindSpore                               | PyTorch          | 功能简述                                                     |
| --------------------------------------- | ---------------- | ------------------------------------------------------------ |
| [Tensor.clip()](./tensor_clip.md)       | Tensor.clamp()   | 裁剪Tensor，约束里面元素的最值                               |
| [Tensor.copy()](./tensor_copy.md)       | Tensor.clone()   | 复制当前Tensor并返回                                         |
| [Tensor.nonzero()](./tensor_nonzero.md) | Tensor.nonzero() | 返回非零元素的坐标。实现无差别，但是在mindspore.ops中也有相同的操作。 |
| [Tensor.split()](./tensor_split.md)     | Tensor.split()   | 分割Tensor，但是Pytorch支持每个分割块有不同的shape           |
| [Tensor.asnumpy()](./tensor2numpy.md)   | Tensor.numpy()   | 将Tensor转换为numpy.ndarray，两者api名称不一样。             |

总体而言，除了Tensor.split()因为传入参数不同，实现的功能可能有点差异，其余api大部分是名称有差异，使用时注意分辨即可。

## mindspore.nn算子

| MindSpore                            | PyTorch     | 简述                                                         |
| ------------------------------------ | ----------- | ------------------------------------------------------------ |
| [BatchNorm2d](./batchnorm2d.md)      | BatchNorm2d | 四维数据归一化。mindspore这边不确定gamma是否就是pytorch对应的weight，beta是否就是对应的bias。 |
| [ReLU](./relu.md)                    | ReLU        | 修正线性单元激活函数。PyTorch可以设置inplace=True，使得输入输出共享一个内存地址。 |
| [ResizeBilinear](./sample_method.md) | Upsample    | 数据采样算子。目前MindSpore没有实现数据上采样的算子，在不构造新算子的前提下，只能用ResizeBilinear算子代替。 |

这些算子因为传入参数或者行为逻辑有差别，最终得到的结果可能会有差异。不过大体上可以满足需求，具体会不会对模型产生重大影响，还需要在模型开发后测试才可知。

## 其他算子

类似nn.Conv2d等的同名算子，在逻辑实现上基本一致的条件下，本文不做赘述。需要注意的是一些nn.Cell初始化权重和偏置的方式可能跟pytorch的不一样，需要注意名称的转换。如在pytorch中，初始化权重经常使用的是kaiming一类的正态分布或者均匀分布，在mindspore中则是称作`HeNormal`和`HeUniform`。在本项目中也有体现。