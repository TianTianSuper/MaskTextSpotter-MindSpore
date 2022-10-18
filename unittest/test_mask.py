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
from src.masktextspotter.mask import *
from src.model_utils.config import config
import pickle as pkl


class TestMask(object):
    @pytest.mark.m
    def test_char2num(self):
        assert char2num('d') == 14
        assert char2num('9') == 10
        assert char2num('we') == 0
    
    @pytest.mark.m
    def test_convert_2d_tuple(self):
        assert convert_2d_tuple(((12,34),(56,78))) == [12,34, 56 ,78]
    
    @pytest.mark.m
    def test_Polygons(self):
        # polygon can get from segmentation
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        img_size = img.shape[-2:]
        po = Polygons(seg[0], img_size, 0)

        assert type(po.transpose(1)) == Polygons
        assert type(po.rotate(15, 'center', 0,0)) == Polygons
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        assert type(po.crop(box_raw[0])) == Polygons
        assert type(po.resize((16,16))) == Polygons
        assert type(po.convert("mask")) == Tensor
        po.set_size((32,32))
        po.get_polygons()
        print(po)
    
    @pytest.mark.m
    def test_char_po(self):
        with open('unittest/case/charbboxs.pkl', 'rb') as f:
            cbox = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        with open('unittest/case/words.pkl', 'rb') as f:
            words = pkl.load(f)
        img_size = img.shape[-2:]
        po = CharPolygons(cbox[1], words[1], use_char_ann=True, size=img_size)

        assert type(po.transpose(1)) == CharPolygons
        assert type(po.rotate(15, 'center', 0,0)) == CharPolygons
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        assert type(po.crop(box_raw[1])) == CharPolygons
        assert type(po.resize((16,16))) == CharPolygons
        po.convert()
        po.convert("seq_char_mask")
        po.set_size((32,32))
        po.creat_color_map(10,2)
        print(po)

    @pytest.mark.m
    def test_SegmentationMask(self):
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        sg = SegmentationMask(seg, img.shape[-2:], 0)
        assert type(sg.transpose(1)) == SegmentationMask
        assert type(sg.rotate(15, 'center', 0,0)) == SegmentationMask
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        assert type(sg.crop(box_raw[1])) == SegmentationMask
        assert type(sg.resize((16,16))) == SegmentationMask
        sg.set_size((32,32))
        print(sg)
        sg.get_polygons()
        assert type(sg.to_np_polygon()) == list
    
    @pytest.mark.m
    def test_SegmentationCharMask(self):
        with open('unittest/case/charbboxs.pkl', 'rb') as f:
            cbox = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        with open('unittest/case/words.pkl', 'rb') as f:
            words = pkl.load(f)
        sg = SegmentationCharMask(cbox, words, size=img.shape[-2:], mode=0)
        assert type(sg.transpose(1)) == SegmentationCharMask
        assert type(sg.rotate(15, 'center', 0,0)) == SegmentationCharMask
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        assert type(sg.crop(box_raw[1])) == SegmentationCharMask
        assert type(sg.resize((16,16))) == SegmentationCharMask
        sg.set_size((32,32))
        print(sg)

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/mask.html', '-m=m'])