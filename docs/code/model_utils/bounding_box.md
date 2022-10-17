# src/model_utils/bounding_box.py

## 主要作用

边界框bounding box的定义与方法，涉及到对数据集ground true的存储和运用。

## 关键模块

`class Boxes`: 边界框

### 最重要的四个变量

- self.bbox = bbox

  Tensor, 存储四个点横纵坐标最值，或者横纵坐标+宽高

- self.size = image_size

  Tensor, 存储图片的shape(宽高)

- self.mode = mode

  bounding box存储数据的属性，与第一点对应

  "xyxy"：存的是最值

  "xywh"：横纵坐标+宽高

- self.extra_fields = {}

  存储必要的变量，如计算后的mask，分数score等。

## 示例

~~~python
box_raw = Tensor([[157., 127., 410., 180.,   0.],
       			 [442., 127., 500., 168.,   1.],
       			 [ 63., 199., 362., 242.,   2.],
      			 [393., 198., 486., 238.,   3.],
      			 [ 71., 270., 381., 311.,   4.]])
shape_raw = Tensor([16,16])
bbox = Boxes(box_raw[:, :4], shape_raw)
~~~

