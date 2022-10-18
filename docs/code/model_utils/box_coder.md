# src/model_utlis/box_coder.py

## 主要作用

对bounding box进行编码和解码操作，参考pytorch的实现

## 模块

`class BoxCoder`: 编码解码器

- `def encode`：对bounding box和proposals编码
- `def decode`：对编码和box解码出预测的边框

## 示例

~~~python
box_coder = BoxCoder(weights=(10., 10., 5., 5.))
~~~

