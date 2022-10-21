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
from src.roi.box_head.feature_extractor import Fpn2Mlp
from src.roi.box_head.head import ROIBoxHead
from src.roi.box_head.inference import PostHandler
from src.roi.box_head.loss import FastRCNNLoss
from src.roi.box_head.predictor import FpnPredict
from src.roi.pooler import Level, Pooler
from src.dataset.generator import NormalManager
from src.roi.roi_combine import CombinedROIHeads
from src.model_utils.config import config

class TestRoiBox(object):
    @pytest.mark.rbh
    def test_fpn2mlp(self):
        # Warning: No enough resource to test
        with open('unittest/case/target.pkl', 'rb') as f:
            target = pkl.load(f)
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        part = Fpn2Mlp(config)

    @pytest.mark.rbh
    def test_ROIBoxHead(self):
        # Warning: No enough resource to test
        with open('unittest/case/target.pkl', 'rb') as f:
            target = pkl.load(f)
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        part = ROIBoxHead(config)

from src.model_utils.matcher import Matcher
from src.roi.mask_head.head import *
from src.roi.mask_head.inference import *
from src.roi.mask_head.loss import *
from src.roi.mask_head.predictor import *
from src.model_utils.config import config

class TestRoiMask(object):
    @pytest.mark.roi
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

    @pytest.mark.roi
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
        
    @pytest.mark.roi
    def test_head(self):
        matcher = Matcher(config.roi.box_head.fg_iou,
                        config.roi.box_head.bg_iou)
        part = ROIMaskHead(config, matcher, (16,16))

class TestRoiPool(object):
    @pytest.mark.roi
    def test_pooler(self):
        data_generator = NormalManager(config)
        img, target = data_generator.generate_single('datasets/icdar2013/train_images', 'datasets/icdar2013/train_gts', '100.jpg')
        shape_raw = img.shape[-2:]
        # bbox = Boxes(box_raw[:, :4], shape_raw, "xywh")
        part = Pooler(64, 64, Tensor([1,1,1,1,1], mstype.float32))

class Testcomb(object):
    @pytest.mark.roi
    def test_comb(self):
        data_generator = NormalManager(config)
        img, target = data_generator.generate_single('datasets/icdar2013/train_images', 'datasets/icdar2013/train_gts', '100.jpg')
        shape_raw = img.shape[-2:]
        # bbox = Boxes(box_raw[:, :4], shape_raw, "xywh")
        part = CombinedROIHeads(config)

if __name__ == '__main__':
    from src.model_utils.config import config
    from mindspore import context
    context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)

    pytest.main(['-vv','-s','--html=unittest/results/roi.html', '-m=roi'])