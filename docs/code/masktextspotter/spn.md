# src/masktextspotter/spn.py

## 主要作用

SPN的主要实现。

## 模块

`class FpnBlock`: 这里定义fpn网络的块，这是构成SEGHead的基本模块。

`class SEGHead`: 利用FpnBlock，这里实现了一个类似U-Net的结构，用来进一步处理特征。

`class SEG`: SPN网络的结构定义。Head -> PostHandle，然后计算loss.