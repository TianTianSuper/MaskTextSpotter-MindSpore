# 比较与torch.Tensor.clamp()的功能差异

## torch.Tensor.clamp()

更多内容详见[torch.Tensor.clamp()](https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html)。

## mindspore.Tensor.clip()

更多内容详见[mindspore.Tensor.clip()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.clip)。

## 使用方式

PyTorch：裁剪Tensor的值，可以指定最大值或者最小值。

MindSpore：裁剪Tensor的值，可以指定最大值或者最小值。与torch实现不一致在于可以指定裁剪后输出数据的类型，设置`dtype`即可。

## 代码示例

```python
# MindSpore
from mindspore import Tensor
from mindspore import dtype as mstype
x = Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype("float32")
y = x.clip(0, 2, mstype.float16)
print(y)
print(y.dtype)
'''
[1. 2. 2. 0. 0. 2. 2. 0.]
Float16
'''

# torch
from torch import Tensor
import numpy as np
x = Tensor([1, 2, 3, -4, 0, 3, 2, 0])
y = x.clamp(0, 2)
print(y)
print(y.dtype)
''' outputs
tensor([1., 2., 2., 0., 0., 2., 2., 0.])
torch.float32
'''
```
