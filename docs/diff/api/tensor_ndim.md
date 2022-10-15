# 比较与torch.Tensor.ndimension()的功能差异

## torch.Tensor.ndimension()

更多内容详见[torch.Tensor.ndimension()](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)。

## mindspore.Tensor.ndim

更多内容详见[mindspore.Tensor.ndim()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.ndim。

## 使用方式

两者实现一样的功能，都返回Tensor维度的数量

## 代码示例

```python
# MindSpore
from mindspore import Tensor
t = Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 5, 5]])
print(t.shape)
print(t.ndim)
'''
(3, 4)
2
'''

# torch
from torch import Tensor
t = torch.Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 5, 5]])
print(t.shape)
print(t.ndimension())
''' outputs
torch.Size([3, 4])
2
'''
```
