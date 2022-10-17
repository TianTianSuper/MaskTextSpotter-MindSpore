# MaskTextSpotter v3 MindSpore 实现

## 概述

本项目旨在使用MindSpore复现MaskTextSpotter v3这个端到端的OCR模型。目前已根据论文和参考代码的逻辑，开发完成部分模块，等待单元测试完成之后，将逐步整合代码。

## 开发规格

### 检测

| Model                                                        | F-measure<br />(Strong Lexicon) | F-measure<br />Week Lexicon) | F-measure<br />Generic Lexicon) |
| ------------------------------------------------------------ | ------------------------------- | ---------------------------- | ------------------------------- |
| [MaskTextSpotter v1](https://github.com/lvpengyuan/masktextspotter.caffe2#models) | 79.3                            | 74.5                         | 64.2                            |
| [MaskTextSpotter v2](https://github.com/MhLiao/MaskTextSpotter) | 82.4                            | 78.1                         | 73.6                            |
| [MaskTextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3) | **83.1**                        | **79.1**                     | **75.1**                        |

### 识别

| Model                                                        | F-measure<br />(Strong Lexicon) | F-measure<br />Week Lexicon) | F-measure<br />Generic Lexicon) |
| ------------------------------------------------------------ | ------------------------------- | ---------------------------- | ------------------------------- |
| [MaskTextSpotter v1](https://github.com/lvpengyuan/masktextspotter.caffe2#models) | 79.3                            | 73.0                         | 62.4                            |
| [MaskTextSpotter v2](https://github.com/MhLiao/MaskTextSpotter) | 83.0                            | 77.7                         | **73.5**                        |
| [MaskTextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3) | **83.3**                        | **78.1**                     | 72.4                            |

### 性能

| Model                                                        | FPS            | Weights                                                      |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| [MaskTextSpotter v1](https://github.com/lvpengyuan/masktextspotter.caffe2#models) | **2.6** (1600) | [Google Drive](https://drive.google.com/open?id=1yPATzUCREBopDIHcsvdYOBB3YpStunMU); [BaiduYun](https://pan.baidu.com/s/1JPZmOQ1LAw98s0GPa-PuuQ) (key: gnpc) |
| [MaskTextSpotter v2](https://github.com/MhLiao/MaskTextSpotter) | 2.0 (1600)     | [Google Drive](https://drive.google.com/open?id=1pPRS7qS_K1keXjSye0kksqhvoyD0SARz) |
| [MaskTextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3) | 2.5 (1440)     | [Google Drive](https://drive.google.com/file/d/1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1fV1RbyQ531IifdKxkScItQ) (key: cnj2) |

注：FPS列中括号内的值表示所输入图像最短边长度。

### 分析

本项目使用MindSpore开发MaskTextSpotter模型的v3规格。由上表可知，虽然在性能指标FPS上，v3稍微比前两个规格落后一点，但无论在字符检测，还是端到端的文本识别任务上，v3规格都有明显的提升。

在Pytorch版本的实现上，v3规格的性能有所下降。但是在本项目的开发过程中，可以尝试结合MindSpore的优势，进行性能调优，查看能否该问题。

## 文件与目录说明

### docs

`docs/diff/api`: 介绍MindSpore与PyTorch实现同样功能算子的差异，多为本项目涉及到的算子。

`docs/code`: 存放每个模块的说明文档，包含代码示例、与pytorch实现的比较等内容。

### src

存放本项目核心模块的实现代码。

`src/dataset`: 读取数据集的方法。包括将原始数据集转换成MindRecord，将MindRecord转换成MindDataset。

`src/masktextspotter`: 实现模型的核心模块。包括backbone(resnet+fpn)、spn、segemetation_mask。

`src/roi`: 实现roi的相关方法，后期将整合进`src/masktextspotter`中。

`src/model_utils`: 模型工具。主要包含边界框的处理、图像读取、部分神经网络层的实现、config参数的管理方法。

`src/general.py`: v3规格主要模型架构。

`src/lr_schedule.py`: 动态学习率实现。

`src/network_define.py`: 定义loss callback，显示必要的训练信息。

### unittest

存放测试本项目模块的测试用例和测试代码。

### default_config.yaml

存放模型需要调用的必要参数，参考maskrcnn在mindspore中的实现所调用config的方法。

### train.py

训练入口。
