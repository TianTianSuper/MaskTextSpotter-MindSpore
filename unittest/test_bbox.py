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
from src.model_utils.config import config
import pickle as pkl

class TestBBox(object):
    @pytest.mark.bbox
    def test_basic(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)

        bbox.add_field("test", 111)
        assert bbox.get_field("test") == 111
        assert bbox.has_field("test") == True
        assert bbox.has_field("testt") == False
        assert len(bbox)
        bbox.area()

    @pytest.mark.bbox
    def test_convert(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw, "xywh")

        bbox.convert("xywh")

    @pytest.mark.bbox
    def test_pack(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)

        with pytest.raises(NameError):
            bbox.pack_field("test")
        print(bbox.bbox)
        bbox.add_field("testt", [157.,127.,410.,180.,112.])
        bbox.pack_field("testt")
    
    @pytest.mark.bbox
    def test_size(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)

        bbox.set_size([16,16])
        bbox.resize([20,20])

    @pytest.mark.bbox
    def test_rotate(self):
        # with open('unittest/case/target.pkl', 'rb') as f:
        #     target = pkl.load(f)
        
        # target.rotate(2, 2, 2, 1)
        pass
        # Warning: due to unfinished test in mask, so delayed
        # including: poly_to_box rotate

    @pytest.mark.bbox
    def test_transpose(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)

        bbox.transpose(0)
        bbox.transpose(1)
    
    @pytest.mark.bbox
    def test_clip_crop(self):
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        shape_raw = img.shape[-2:]
        bbox = Boxes(box_raw[:, :4], shape_raw)

        bbox.clip_to_image()
        # warning: crop need to be tested after seg_mask
        # bbox.crop(box_raw[:,:2])

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/bbox.html', '-m=bbox'])