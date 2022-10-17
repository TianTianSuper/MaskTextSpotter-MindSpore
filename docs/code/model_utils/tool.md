# src/model_utils/tool.py

## 主要作用

参考pytorch，主要包含的是所谓的“安全拼接”，即在传入数据前，判断是否是列表或者元组类型的，如果是就去掉。但是在测试过程中，发现mindspore已经将拼接传入的数据类型固定了，而这里面并没有列表和元组。他俩的作用仅仅只是框住需要拼接的数据而已。这个模块后期考虑优化或者删除。

## 模块

`class SafeConcat`: 安全拼接

## 示例

~~~python
nm = SafeConcat()
t1 = Tensor([1,2,3,4],mstype.float32)
t2 = Tensor([2.,3.,4.,5.],mstype.float32)
cat = nm((t1,t2))
~~~

