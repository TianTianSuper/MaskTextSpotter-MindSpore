# src/model_utils/bbox_ops.py

## 主要作用

针对bounding box的操作集合，基本参考了pytorch的实现。

## 重要方法

`def boxlist_nms`: 给bbox赋分

`def remove_small_boxes`: 移除规定大小下的bbox

`def boxlist_iou`: 计算两个边框的iou，必须是xyxy模式

`def cat_boxlist(bboxes)`: 合并bbox，要求输入必须是元组或者列表

## 示例

~~~python
box_raw = Tensor([[157., 127., 410., 180.,   0.],
       			 [442., 127., 500., 168.,   1.],
       			 [ 63., 199., 362., 242.,   2.],
      			 [393., 198., 486., 238.,   3.],
      			 [ 71., 270., 381., 311.,   4.]])
shape_raw = Tensor([16,16])
bbox = Boxes(box_raw[:, :4], shape_raw)

nms_box = boxlist_nms(bbox, 0.9)
new_box = remove_small_boxes(bbox, 2)
iou = boxlist_iou(bbox,bbox)
con_box = cat_boxlist((bbox, bbox, bbox))
~~~

