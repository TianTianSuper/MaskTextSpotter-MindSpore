# src/lr_schedule.py

## 主要作用

实现动态学习率。虽然mindspore有相关的类，但是在现阶段的实现上，还是自定义一个方法然后返回学习率的tensor来得方便。同时参考了maskrcnn(mindspore)和pytorch的实现。另外有一个问题就是要确定更新的步数，目前参考的是pytorch版本的。

## 模块

`def warmup_lr(config, rank_size=1, start_steps=0)`