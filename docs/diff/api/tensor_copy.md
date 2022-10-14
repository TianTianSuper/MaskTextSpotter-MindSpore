# 比较与torch.Tensor.clone()的功能差异

## torch.Tensor.clone()

更多内容详见[torch.Tensor.clone()](https://pytorch.org/docs/stable/generated/torch.Tensor.clone.html)。

## mindspore.Tensor.copy()

更多内容详见[mindspore.Tensor.copy()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.copy)。

## 使用方式

PyTorch：复制一个Tensor并返回。

MindSpore：直接调用这个方法，就可以复制并返回对应的Tensor。

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
