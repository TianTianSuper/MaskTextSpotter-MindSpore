# src/general.py

## 主要作用

定义masktextspotter的结构，整合所有相关的模块于一体去计算。

## 模块

`class GeneralLoss(nn.Cell)`：loss的计算Cell。因为在论文中，即使部分loss有scale，但是都设置成了1。目前主要搜集所有的loss然后相加。

`class MaskTextSpotter3(nn.Cell)`：Masktextspotter模型的主要结构定义，整合了先前的代码。

## 示例

~~~python
net = MaskTextSpotter3(config)
net.set_train(True)
model = Model(net)
model.train('''...''')
~~~

