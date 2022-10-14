# 比较与torch.Tensor.split()的功能差异

## torch.Tensor.split()

~~~python
Tensor.split(split_size, dim=0)
~~~

更多内容详见[torch.Tensor.split()](https://pytorch.org/docs/stable/generated/torch.Tensor.split.html)。

## mindspore.Tensor.split()

```python
split(axis=0, output_num=1)
```

更多内容详见[mindspore.Tensor.split()](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.split)。

## 使用方式

PyTorch：分割Tensor，既可以分割成相同的shape，也可以传入一个list，分别指定每个分割块中维度的元素数量

MindSpore：根据指定的轴和分割数量对Tensor进行分割，会被分割成相同的shape。

torch实现中的`dim`与mindspore实现中的`axis`意义是一样的，都是表示在特定的维度上做切割。两者主要的差异在于分割的方式。PyTorch支持指定每一个分割块的元素数量，即`split_size`可以传入list来定义，也可以传入int，将Tendor等分；但是MindSpore里面的`output_num`只是等分块的数量。

## 代码示例

```python
# MindSpore
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]), mstype.int32)
print(x)
'''
[[1 1 1 1]
 [2 2 2 2]]
'''
output = x.split(1, 2)
print(output)
'''
(Tensor(shape=[2, 2], dtype=Int32, value=
[[1, 1],
 [2, 2]]), Tensor(shape=[2, 2], dtype=Int32, value=
[[1, 1],
 [2, 2]]))
'''

# torch
import numpy as np
from torch import Tensor
x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]).astype("int32"))
print(x)
'''
tensor([[1., 1., 1., 1.],
        [2., 2., 2., 2.]])
'''
output = x.split(2,1)
print(output)
'''
(tensor([[1., 1.],
        [2., 2.]]), tensor([[1., 1.],
        [2., 2.]]))
'''
output = x.split([1, 3],1)
print(output)
'''
(tensor([[1.],
        [2.]]), tensor([[1., 1., 1.],
        [2., 2., 2.]]))
'''
```
