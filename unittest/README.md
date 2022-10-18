# Unit Test 单元测试

## 各模块进度

| 模块          | 路径                                     | 状态               | 描述                                                         |
| ------------- | ---------------------------------------- | ------------------ | ------------------------------------------------------------ |
| 数据处理      | `src/dataset/generator.py`               | :white_check_mark: |                                                              |
| backbone      | `src/masktextspotter/fpn_neck.py`        | :white_check_mark: |                                                              |
| backbone      | `src/masktextspotter/resnet50.py`        | :white_check_mark: |                                                              |
| proposal(SPN) | `src/masktextspotter/inference.py`       | :black_circle:     |                                                              |
| proposal(SPN) | `src/masktextspotter/loss.py`            | :black_circle:     |                                                              |
| proposal(SPN) | `src/masktextspotter/spn.py`             | :black_circle:     |                                                              |
| seg_mask      | `src/masktextspotter/mask.py`            | :white_check_mark: |                                                              |
| roi           | `src/roi/pooler.py`                      | :black_circle:     |                                                              |
| roi           | `src/roi/roi_align.py`                   | :black_circle:     |                                                              |
| roi           | `src/roi/roi_combine.py`                 | :black_circle:     |                                                              |
| roi           | `src/roi/box_head/feature_extractor.py`  | :black_circle:     |                                                              |
| roi           | `src/roi/box_head/head.py`               | :black_circle:     |                                                              |
| roi           | `src/roi/box_head/inference.py`          | :black_circle:     |                                                              |
| roi           | `src/roi/box_head/loss.py`               | :black_circle:     |                                                              |
| roi           | `src/roi/box_head/predictor.py`          | :black_circle:     |                                                              |
| roi           | `src/roi/mask_head/feature_extractor.py` | :black_circle:     |                                                              |
| roi           | `src/roi/mask_head/head.py`              | :black_circle:     |                                                              |
| roi           | `src/roi/mask_head/inference.py`         | :black_circle:     |                                                              |
| roi           | `src/roi/mask_head/loss.py`              | :black_circle:     |                                                              |
| roi           | `src/roi/mask_head/predictor.py`         | :black_circle:     |                                                              |
| utils         | `src/model_utils/assigner.py`            | :no_entry_sign:    | 未使用                                                       |
| utils (bbox)  | `src/model_utils/bbox_ops.py`            | :black_circle:     |                                                              |
| utils (bbox)  | `src/model_utils/bounding_box.py`        | :warning:          | 'crop' and 'rotate' should be tested after testing seg_mask  |
| utils (bbox)  | `src/model_utils/box_coder.py`           | :white_check_mark: |                                                              |
| utils         | `src/model_utils/images.py`              | :white_check_mark: |                                                              |
| utils         | `src/model_utils/config.py`              | :no_entry_sign:    | implemented from maskrcnn(mindspore)<br>and run in good status |
| utils         | `src/model_utils/local_adapter.py`       | :no_entry_sign:    | implemented from maskrcnn(mindspore)                         |
| utils (layer) | `src/model_utils/blocks.py`              | :white_check_mark: |                                                              |
| utils (layer) | `src/model_utils/norm.py`                | :white_check_mark: |                                                              |
| utils         | `src/model_utils/sampler.py`             | :no_entry_sign:    | 未使用                                                       |
| utils         | `src/model_utils/tools.py`               | :white_check_mark: |                                                              |
| main          | `src/general.py`                         | :black_circle:     |                                                              |
| main          | `src/lr_schedule.py`                     | :black_circle:     |                                                              |
| main          | `src/network_define.py`                  | :black_circle:     |                                                              |
| train         | `train.py`                               | :black_circle:     |                                                              |

进度状态定义

| 图标               | 说明                         |
| ------------------ | ---------------------------- |
| :white_check_mark: | 通过所有测试用例             |
| :x:                | 测试用例不通过               |
| :warning:          | 通过测试用例，但是存在疑问   |
| :no_entry_sign:    | 因为某些原因跳过该模块的测试 |
| :black_circle:     | 未开始                       |

