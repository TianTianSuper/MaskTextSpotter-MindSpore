# src/dataset/generator.py

## 主要作用

生成MindSpore支持的数据集，方便模型处理。

## 模块

`class DatasetsManager`: 数据集生成器

- `def init_mindrecords`: 读入原始数据集，生成mindrecord格式的数据记录到磁盘中。目前支持的数据集有`icdar2013`、`icdar2015`和`scut-eng-char`。

- `def __init_dataset`: 将mindrecord处理成MindDataset。不直接处理原始数据的原因是mindrecord和minddataset都支持并行处理，能够一次处理多条数据记录。其中mindrecord可以在init的时候指定，minddataset的map方法也可以并行处理优化的方法，提升数据预处理的性能。

- `def char2num`: 将英文字符转换成数字。

- `def __gt2boxes(self, gt)`: 提取原始数据集的ground true，包括边框的点，边框内所表示的单词或字符。

- `def __restore_dataset(self, image, ground_trues)`: 转换mindrecord记录到minddataset。此处的输入均是二进制记录，需要预先解码。而后将image转换成RGB编码，ground_true转换成多个输出，包括：

  - words: 整个单词
  - boxes: 框住所有字符的最大边框，只需存储x,y的最大最小值即可。*这里好像存的是索引，因为都-1(?)*
  - charsboxes: 每个单词字符的边界框，另外还添加了字符的数字编码，还有所在单词的单词索引
  - segmentations: 每个边界框的记录
  - labels: 标签，*1或者-1*(?)

  以上都会被封装在Boxes类中，名为target。其中也保存了经过seg_mask计算后的单词和字符mask，分别称作"mask"和"char_mask"。

## 示例

~~~python
dm = DatasetsManager(config=config) # 定义类
dm.init_mindrecords() # 初始化records
ds = dm.init_dataset() # 初始化数据集
~~~

