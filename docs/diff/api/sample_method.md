# 比较与torch.nn.Upsample()的功能差异

## torch.Tensor.clamp()

~~~python
torch.nn.Upsample(
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None
)(input)
~~~

更多内容详见[[torch.nn.Upsample](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Upsample)。

## mindspore.nn.ResizeBilinear()

~~~python
class mindspore.nn.ResizeBilinear()(x, size=None, scale_factor=None, align_corners=False)
~~~

更多内容详见[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)。

这个差异已经在[官方的文档](https://mindspore.cn/docs/zh-CN/r1.8/note/api_mapping/pytorch_diff/ResizeBilinear.html)里面说明了，两者功能确实相差较大。如果要实现upsample，目前mindspore还没有现成的算子支持。在本项目的实现中，暂时先由mindspore.nn.ResizeBilinear()代替upsample的采样操作。
