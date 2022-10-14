# 比较与torch.nn.Relu的功能差异

## torch.nn.ReLU

~~~python
torch.nn.ReLU(inplace=False)(x)
~~~

更多内容详见[torch.nn.Relu](https://pytorch.org/docs/1.5.0/nn.html?highlight=relu#torch.nn.ReLU)。

## mindspore.nn.ReLU

```python
mindspore.nn.ReLU()(x)
```

更多内容详见[mindspore.nn.ReLU](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.ReLU.html?highlight=relu)。

## 使用方式

PyTorch：可以指定inplace，若为True，则替换掉输入对象的值。参考[CSDN](https://blog.csdn.net/manmanking/article/details/104830822)。

MindSpore：无inplace参数，传入的参数只有Tensor。

## 遇到的问题

**Warning：官方源代码存在某些ReLU操作是加了inplace=True**

这个问题暂时影响不大，因为在一个CellList中，目的是顺序计算求得最终的结果。inplace=True只是在性能上可能会提升一点，因为输出的结果与输入的结果共享同一个地址，避免了反复申请和释放内存的时空消耗。
