# src/masktextspotter/mask.py

## 主要作用

加mask的操作，主要属于bounding box数据的转换和处理，这个非常重要

## 重要模块

`class SegmentationMask`: 针对处理一整张图片的bbox信息

`class Polygons`: 针对处理一整张图片的一项bbox信息(?)

`class SegmentationCharMask`: 针对处理字符的bbox信息

`class SegmentationMask`: 针对处理一字符的bbox信息(?)



## 示例

~~~python
po = Polygons([111.,222.,33.,44.,55.,22.,44.,77.], [16,16], 0)
~~~

