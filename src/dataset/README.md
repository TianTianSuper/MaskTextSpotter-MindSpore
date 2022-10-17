# 数据集处理

## 论文与官方源码采用的数据集

`icdar-2013`

`icdar-2015`

`synthtext`

`total-text`

`scut-eng-char`

### ground_true 组织方式

|      | icdar-2013 | icdar-2015 | synthtext | total-text | scut-eng-char |
| ---- | ---------- | ------------ | ----------- | ---------- | --------------- |
| 英文单词选框 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 单一字符选框 | :white_check_mark: | :x: | :white_check_mark: | :x: | :white_check_mark: |
| 一个单词一行数据 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 组织在train_gts中 | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: |

## 处理说明

在源码实现中，如果从0开始训练，则会混合上述五个数据集，一起训练。而对于不同数据集，处理的方式也不一样，但是最后都要达到整合的目标。而对于有预训练模型的训练，则可以只输入一个数据集，并且继续训练下去。

从简便开始，本项目首先根据`icdar-2013`，构建`mindrecord`，然后再将其转换为mindspore的数据集格式，在转换时，通过`map`函数再对候选框做进一步的转换，如转换成bounding_box。

原有生成mindrecord的处理已经合并到generator中。
