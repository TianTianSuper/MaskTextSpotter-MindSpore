# Unit Test 单元测试

## 各模块进度

| 模块          | 路径                               | 状态               | 描述                                                         |
| ------------- | ---------------------------------- | ------------------ | ------------------------------------------------------------ |
| 数据处理      | `src/dataset/generator.py`         | :white_check_mark: |                                                              |
| backbone      | `src/masktextspotter/fpn_neck.py`  | :white_check_mark: |                                                              |
| backbone      | `src/masktextspotter/resnet50.py`  | :white_check_mark: |                                                              |
| proposal(SPN) | `src/masktextspotter/inference.py` | :white_check_mark: |                                                              |
| proposal(SPN) | `src/masktextspotter/loss.py`      | :white_check_mark: |                                                              |
| proposal(SPN) | `src/masktextspotter/spn.py`       | :white_check_mark: |                                                              |
| seg_mask      | `src/masktextspotter/mask.py`      | :white_check_mark: |                                                              |
| roi           | `src/roi`                          | :white_check_mark: |                                                              |
| utils         | `src/model_utils/assigner.py`      | :white_circle:     | no use                                                       |
| utils (bbox)  | `src/model_utils/bbox_ops.py`      | :white_check_mark: |                                                              |
| utils (bbox)  | `src/model_utils/bounding_box.py`  | :white_check_mark: |                                                              |
| utils (bbox)  | `src/model_utils/box_coder.py`     | :white_check_mark: |                                                              |
| utils         | `src/model_utils/images.py`        | :white_check_mark: |                                                              |
| utils         | `src/model_utils/config.py`        | :white_circle:     | implemented from maskrcnn(mindspore) and run in good status  |
| utils         | `src/model_utils/local_adapter.py` | :white_circle:     | implemented from maskrcnn(mindspore) and run in good status with `train.py` |
| utils (layer) | `src/model_utils/blocks.py`        | :white_check_mark: |                                                              |
| utils (layer) | `src/model_utils/norm.py`          | :white_check_mark: |                                                              |
| utils         | `src/model_utils/sampler.py`       | :white_check_mark: |                                                              |
| utils         | `src/model_utils/tools.py`         | :white_check_mark: |                                                              |
| main          | `src/general.py`                   | :white_check_mark: |                                                              |
| main          | `src/lr_schedule.py`               | :white_check_mark: |                                                              |
| main          | `src/network_define.py`            | :white_check_mark: |                                                              |
| train         | `train.py`                         | :white_check_mark: | passed init                                                  |

进度状态定义

| 图标               | 说明                         |
| ------------------ | ---------------------------- |
| :white_check_mark: | 通过所有测试用例             |
| :x:                | 测试用例不通过               |
| :warning:          | 通过测试用例，但是存在疑问   |
| :white_circle:     | 因为某些原因跳过该模块的测试 |
| :black_circle:     | 未开始                       |

