# 比较与torch.nn.BatchNorm2d的功能差异

已有的官方文档链接：https://www.mindspore.cn/docs/zh-CN/r1.8/note/api_mapping/pytorch_diff/BatchNorm2d.html

## torch.nn.BatchNorm2d

~~~python
class torch.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
~~~

更多内容详见[torch.nn.BatchNorm2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.BatchNorm2d)。

## mindspore.Tensor.split()

```python
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.9,
    affine=True,
    gamma_init="ones",
    beta_init="zeros",
    moving_mean_init="zeros",
    moving_var_init="ones",
    use_batch_statistics=None,
    data_format="NCHW"
)
```

更多内容详见[mindspore.nn.BatchNorm2d](https://mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.BatchNorm2d.html#mindspore.nn.BatchNorm2d)。

## 差异点

参考官方文档中的“使用方式”一栏。

## 遇到的问题

**无法初始化权重和偏置**

在pytorch中，有weight和bias，但是在mindspore没有，但是有gamma和beta的初始化操作。尚未确定两者是否有关联。
