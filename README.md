# MaskTextSpotter v3 MindSpore 实现

## 概述

本项目旨在使用MindSpore复现MaskTextSpotter v3这个端到端的OCR模型。目前已根据论文和参考代码的逻辑，开发完成部分模块，等待单元测试完成之后，将逐步整合代码。

## 文件与目录说明

### docs

`docs/diff/api`: 介绍MindSpore与PyTorch实现同样功能算子的差异，多为本项目涉及到的算子。

`docs/diff/code`: 计划存放本项目代码与作者源代码的差异，或将迁移至`notebook.md`。

### src

存放本项目核心模块的实现代码。

### unittest

存放测试本项目模块的测试用例和测试代码。

### default_config.yaml

存放模型需要调用的必要参数，参考maskrcnn在mindspore中的实现所调用config的方法。
