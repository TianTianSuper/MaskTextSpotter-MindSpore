# 随记Handbook

## pre_train 与 mix_train 的区别

- pre_train：一次只训练一个数据集
- mix_train：将多个数据集里面的图像混合训练

需要非常注意的是：在torch的实现过程中，如果只是单纯测试某个模块，那么调用到的就只是config的默认参数，跟yaml定义的有很大区别！如果要测试单个模块，记得先找到yaml对应的参数，然后传入模块中，或者直接修改默认的config。pytorch这样写太容易将参数混淆了。

基于两者的区别，在MindSpore实现中定义命名规则如下：
- single_train：一次只训练一个数据集
- multi_train：多个数据集混合训练

经过winmerge的比对，其实两套yaml参数只是在数据集（含训练集、测试集）上面的设置有区别。

