# 比较与torch.Tensor.numpy()的功能差异

## torch.Tensor.numpy()

更多内容详见[torch.Tensor.numpy()](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)。

## mindspore.Tensor.asnumpy()

更多内容详见[mindspore.Tensor.asnumpy()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.asnumpy)。

## 使用方式

PyTorch：将Tensor转换为numpy.ndarray，两者共享内存地址，从而在Tensor上的修改会反映到ndarray上。

MindSpore：功能与torch实现的版本完全一致。

## 代码示例

```python
# MindSpore
from mindspore import Tensor
import numpy as np
x = Tensor(np.array([1, 2], dtype=np.float32))
y = x.asnumpy()
y[0] = 11
print(x)
print(type(y))
'''
[11.  2.]
<class 'numpy.ndarray'>
'''

# torch
from torch import Tensor
import numpy as np
x = Tensor(np.array([1, 2], dtype=np.float32))
y = x.numpy()
y[0] = 11
print(x)
print(type(y))
''' outputs
tensor([11.,  2.])
<class 'numpy.ndarray'>
'''
```
