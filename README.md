# MaskTextSpotter v3 MindSpore 实现

## 概述

本项目旨在使用MindSpore复现MaskTextSpotter v3这个端到端的OCR模型。目前已根据论文和参考代码的逻辑，开发完成部分模块，等待单元测试完成之后，将逐步整合代码。

## 前期任务进度

| 分任务       | 描述                                                         | 进度                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 支持分析     | MindSpore支持分析（算子/API）等支持分析文档，<br>需包括与Pytorch API差异点（接口名称/接口参数等不同） | :white_check_mark: <a href=https://github.com/TianTianSuper/MaskTextSpotter-MindSpore/tree/master/docs/diff/api>支持分析文档</a> |
| 开发规格确认 | 根据论文实验描述/业界pytorch仓规格支持情况，<br>确认需开发的模型规则 | :white_check_mark: <a href=https://github.com/TianTianSuper/MaskTextSpotter-MindSpore#%E5%BC%80%E5%8F%91%E8%A7%84%E6%A0%BC>开发规格</a> |
| 建仓         | github建立模型的repo，记录模型开发的过程                     | :white_check_mark: 本公开repo                                |
| 代码完成     | 数据处理、模型定义、模型训练，<br>各部分需完成unit test；<br>完成各模块的文档及代码示例说明和心得 | :white_check_mark: <a href=src/dataset>数据处理</a><br>:white_check_mark: <a href=src/>模型定义</a><br>:white_check_mark: <a href=train.py>模型训练</a><br>:ballot_box_with_check: <a href=unittest>unit test</a><br>:white_check_mark: <a href=docs/code>代码示例</a><br>:white_check_mark: <a href=docs/feelings.md>心得</a> |



## 开发规格

### 检测

| Model                                                        | F-measure<br />(Strong Lexicon) | F-measure<br />(Week Lexicon) | F-measure<br />(Generic Lexicon) |
| ------------------------------------------------------------ | ------------------------------- | ----------------------------- | -------------------------------- |
| [MaskTextSpotter v1](https://github.com/lvpengyuan/masktextspotter.caffe2#models) | 79.3                            | 74.5                          | 64.2                             |
| [MaskTextSpotter v2](https://github.com/MhLiao/MaskTextSpotter) | 82.4                            | 78.1                          | 73.6                             |
| [MaskTextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3) | **83.1**                        | **79.1**                      | **75.1**                         |

### 识别

| Model                                                        | F-measure<br />(Strong Lexicon) | F-measure<br />(Week Lexicon) | F-measure<br />(Generic Lexicon) |
| ------------------------------------------------------------ | ------------------------------- | ----------------------------- | -------------------------------- |
| [MaskTextSpotter v1](https://github.com/lvpengyuan/masktextspotter.caffe2#models) | 79.3                            | 73.0                          | 62.4                             |
| [MaskTextSpotter v2](https://github.com/MhLiao/MaskTextSpotter) | 83.0                            | 77.7                          | **73.5**                         |
| [MaskTextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3) | **83.3**                        | **78.1**                      | 72.4                             |

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

## 参数设置说明

关于参数的设置，作者源码中给了两套方案：pre_train和mix_train。

- pre_train：一次只训练一个数据集
- mix_train：将多个数据集里面的图像混合训练

需要非常注意的是：在torch的实现过程中，如果只是单纯测试某个模块，那么调用到的就只是config的默认参数，跟yaml定义的有很大区别。如果要测试单个模块，记得先找到yaml对应的参数，然后再看default的参数，或者直接修改默认的config。因为yaml是会覆盖default的config的。

另外这两套参数只是在数据集的相关参数上不一致，大部分涉及到模型的参数时一样的。这说明只需留意上述问题和数据集参数即可，不用担心两者搭建的模型参数不一致。
