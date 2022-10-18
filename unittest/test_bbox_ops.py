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
from src.model_utils.bounding_box import Boxes
from src.model_utils.bbox_ops import *
from src.model_utils.config import config
import pickle as pkl

class TestBBoxp(object):
    @pytest.mark.bops
    def test_nms(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)
        bbox.add_field("score", Tensor([1,1,1,1,1],mstype.float32))

        assert type(boxlist_nms(bbox, 0.9)) == type(bbox)

    @pytest.mark.bops
    def test_remove_small_boxes(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)
        remove_small_boxes(bbox, 2)
        with pytest.raises(TypeError):
            remove_small_boxes(bbox, (2,2))

    @pytest.mark.bops
    def test_boxlist_iou(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)
        boxlist_iou(bbox,bbox)

    @pytest.mark.bops
    def test_cat_boxlist(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)
        cat_boxlist((bbox, bbox, bbox))

    @pytest.mark.bops
    def test_cat_boxlist(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)
        cat_boxlist_gt((bbox, bbox, bbox))

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/bbox_ops.html', '-m=bops'])