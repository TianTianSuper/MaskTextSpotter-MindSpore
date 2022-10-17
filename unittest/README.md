# Unit Test 单元测试

## 各模块进度

| 模块          | 路径                                     | 状态                       |
| ------------- | ---------------------------------------- | -------------------------- |
| 数据处理      | `src/dataset/generator.py`               | :white_check_mark:         |
| backbone      | `src/masktextspotter/fpn_neck.py`        | :ballot_box_with_check:    |
| backbone      | `src/masktextspotter/resnet50.py`        | :ballot_box_with_check:    |
| proposal(SPN) | `src/masktextspotter/inference.py`       | :black_circle:             |
| proposal(SPN) | `src/masktextspotter/loss.py`            | :black_circle:             |
| proposal(SPN) | `src/masktextspotter/spn.py`             | :black_circle:             |
| seg_mask      | `src/masktextspotter/mask.py`            | :black_circle:             |
| roi           | `src/roi/pooler.py`                      | :black_circle:             |
| roi           | `src/roi/roi_align.py`                   | :black_circle:             |
| roi           | `src/roi/roi_combine.py`                 | :black_circle:             |
| roi           | `src/roi/box_head/feature_extractor.py`  | :black_circle:             |
| roi           | `src/roi/box_head/head.py`               | :black_circle:             |
| roi           | `src/roi/box_head/inference.py`          | :black_circle:             |
| roi           | `src/roi/box_head/loss.py`               | :black_circle:             |
| roi           | `src/roi/box_head/predictor.py`          | :black_circle:             |
| roi           | `src/roi/mask_head/feature_extractor.py` | :black_circle:             |
| roi           | `src/roi/mask_head/head.py`              | :black_circle:             |
| roi           | `src/roi/mask_head/inference.py`         | :black_circle:             |
| roi           | `src/roi/mask_head/loss.py`              | :black_circle:             |
| roi           | `src/roi/mask_head/predictor.py`         | :black_circle:             |
| utils         | `src/model_utils/assigner.py`            | :no_entry_sign: 模块未使用 |
| utils (bbox)  | `src/model_utils/bbox_ops.py`            | :black_circle:             |
| utils (bbox)  | `src/model_utils/bounding_box.py`        | :black_circle:             |
| utils (bbox)  | `src/model_utils/box_coder.py`           | :black_circle:             |
| utils         | `src/model_utils/images.py`              | :black_circle:             |
| utils         | `src/model_utils/config.py`              | :black_circle:             |
| utils         | `src/model_utils/local_adapter.py`       | :black_circle:             |
| utils (layer) | `src/model_utils/blocks.py`              | :black_circle:             |
| utils (layer) | `src/model_utils/norm.py`                | :black_circle:             |
| utils         | `src/model_utils/sampler.py`             | :black_circle:             |
| utils         | `src/model_utils/tools.py`               | :black_circle:             |
| main          | `src/general.py`                         | :black_circle:             |
| main          | `src/lr_schedule.py`                     | :black_circle:             |
| main          | `src/network_define.py`                  | :black_circle:             |
| train         | `train.py`                               | :ballot_box_with_check:    |

进度状态定义

| 图标                    | 说明                                   |
| ----------------------- | -------------------------------------- |
| :white_check_mark:      | 通过所有测试用例                       |
| :ballot_box_with_check: | 通过所有测试用例，未检验作为整体的运行 |
| :x:                     | 测试用例不通过                         |
| :warning:               | 通过测试用例，但是存在疑问             |
| :no_entry_sign:         | 因为某些原因取消该模块的测试           |
| :black_circle:          | 未开始                                 |
