# MaskTextSpotter v3 MindSpore 实现

## 概述

本项目旨在使用MindSpore复现MaskTextSpotter v3这个端到端的OCR模型。目前已根据论文和参考代码的逻辑，开发完成部分模块，等待单元测试完成之后，将逐步整合代码。

## 文件与目录说明

### docs

`docs/diff/api`: 介绍MindSpore与PyTorch实现同样功能算子的差异，多为本项目涉及到的算子。

`docs/code`: 存放每个模块的说明文档，包含代码示例、与pytorch实现的比较等内容。

### src

存放本项目核心模块的实现代码。

### unittest

存放测试本项目模块的测试用例和测试代码。

### default_config.yaml

存放模型需要调用的必要参数，参考maskrcnn在mindspore中的实现所调用config的方法。
