import os
import sys

sys.path.append(os.getcwd())

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import normal
from mindspore.common.initializer import Normal
from src.roi.pooler import Level, Pooler
from src.model_utils.bounding_box import Boxes
from src.model_utils.config import config
from src.dataset.generator import NormalManager
import pickle as pkl


class TestRoiPool(object):
    @pytest.mark.p
    def test_pooler(self):
        data_generator = NormalManager(config)
        img, target = data_generator.generate_single('datasets/icdar2013/train_images', 'datasets/icdar2013/train_gts', '100.jpg')
        shape_raw = img.shape[-2:]
        # bbox = Boxes(box_raw[:, :4], shape_raw, "xywh")
        part = Pooler(64, 64, Tensor([1,1,1,1,1], mstype.float32))

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/roi_pooler.html', '-m=p'])