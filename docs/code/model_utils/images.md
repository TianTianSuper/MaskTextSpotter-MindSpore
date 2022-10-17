# src/model_utlis/images.py

## 主要作用

处理一批图像数据，并且存储。

## 模块

`class Images`: 图像的集合类

- self.tensors：图像的Tensor表示，通常有多个Tensor，往后生成这个类的方法不允许只输入一个tensor，默认的类型是float32
- self.image_sizes：图像大小（集合）

`def to_image_list`: 生成Images类，不允许只使用一张图片初始化这个类

## 示例

~~~python
nm = Images(img, img.shape[-2:])

# --------------------------------

to_image_list(Tensor([img, img],mstype.float32))
to_image_list([Tensor(img,mstype.float32),Tensor(img,mstype.float32)])
to_image_list((Tensor(img,mstype.float32),Tensor(img,mstype.float32)))
~~~

