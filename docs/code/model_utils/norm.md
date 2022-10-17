# src/model_utils/norm.py

## 主要作用

改编自pytorch的FrozenBatchNorm，主要应用于backbone的resnet中，是resnet50的一个特性

## 模块

`class FixedBatchNorm2d`: 聚合了卷积层、batchnorm和relu

## 示例

~~~python
block = FixedBatchNorm2d(256)
out = block(Tensor(np.ones(256), mindspore.float32))
~~~

