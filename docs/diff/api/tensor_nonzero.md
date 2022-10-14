# 比较与torch.Tensor.nonzero()的功能差异

## torch.Tensor.nonzero()

更多内容详见[torch.Tensor.nonzero()](https://pytorch.org/docs/stable/generated/torch.Tensor.nonzero.html)。

## mindspore.Tensor.nonzero()

更多内容详见[mindspore.Tensor.nonzero()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.nonzero)。

## 使用方式

PyTorch：计算并返回Tensor非零元素的坐标，返回的是LongTensor，行为与mindspore实现的一致。

MindSpore：计算Tensor中非零元素的下标，返回的数据类型是int64。

## 代码示例

```python
# MindSpore
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
x = Tensor(np.array([[[1,  0], [-5, 0]]]), mstype.int32)
output = x.nonzero()
print(output)
print(output.dtype)
'''
[[0 0 0]
 [0 1 0]]
Int64
'''

# torch
from torch import Tensor
import numpy as np
a = Tensor(np.array([[[1,  0], [-5, 0]]]).astype("int32"))
output = a.nonzero()
print(output)
''' outputs
tensor([[0, 0, 0],
        [0, 1, 0]])
'''
```
