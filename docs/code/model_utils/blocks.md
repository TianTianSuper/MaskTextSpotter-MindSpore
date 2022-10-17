# src/model_utils/blocks.py

## 主要作用

组合多个cell成一个block

## 模块

`class ConvBnReluBlock`: 聚合了卷积层、batchnorm和relu

## 示例

~~~python
block = ConvBnReluBlock(256,256)
out = block(Tensor(np.ones(256,256,3,3), mindspore.float32))
~~~

