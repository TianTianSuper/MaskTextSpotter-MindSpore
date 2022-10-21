import os
import sys
import mindspore
sys.path.append(os.getcwd())

import pytest
import pickle as pkl
import numpy as np
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import normal
from mindspore.common.initializer import Normal
from src.model_utils.matcher import Matcher
from src.roi.mask_head.head import *
from src.roi.mask_head.inference import *
from src.roi.mask_head.loss import *
from src.roi.mask_head.predictor import *
from src.model_utils.config import config

class TestRoiMask(object):
    @pytest.mark.rm
    def test_project_char_masks_on_boxes(self):

        # with open('unittest/case/target.pkl', 'rb') as f:
        #     target = pkl.load(f)
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/charbboxs.pkl', 'rb') as f:
            charbbox = pkl.load(f)
        from src.dataset.generator import NormalManager
        data_generator = NormalManager(config)
        img, target = data_generator.generate_single('datasets/icdar2013/train_images', 'datasets/icdar2013/train_gts', '100.jpg')
        
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        part = project_char_masks_on_boxes(target.get_field("masks"),target.get_field("char_masks"),target,(16,16))

    @pytest.mark.rm
    def test_keep_only_positive_boxes(self):
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/charbboxs.pkl', 'rb') as f:
            charbbox = pkl.load(f)
        from src.dataset.generator import NormalManager
        data_generator = NormalManager(config)
        img, target = data_generator.generate_single('datasets/icdar2013/train_images', 'datasets/icdar2013/train_gts', '100.jpg')
        
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        with pytest.raises(AssertionError):
            part = keep_only_positive_boxes(target, 32)
        part = keep_only_positive_boxes([target], 1)
        part = keep_only_positive_boxes([target], 32)
        
    @pytest.mark.rm
    def test_head(self):
        matcher = Matcher(config.roi.box_head.fg_iou,
                        config.roi.box_head.bg_iou)
        part = ROIMaskHead(config, matcher, (16,16))


if __name__ == '__main__':
    from src.model_utils.config import config
    from mindspore import context
    context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)

    pytest.main(['-vv','-s','--html=unittest/results/roi_box_head.html', '-m=rm'])