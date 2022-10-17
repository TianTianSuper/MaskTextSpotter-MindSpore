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
from src.model_utils.images import *
from src.model_utils.config import config
import pickle as pkl


class TestImg(object):
    @pytest.mark.img
    def test_images(self):
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        nm = Images(img, img.shape[-2:])
        assert nm.get_sizes() == img.shape[-2:]

    @pytest.mark.img
    def test_tols(self):
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
            print(img)
        to_image_list(Tensor([img, img],mstype.float32))
        with pytest.raises(AssertionError):
            to_image_list(Tensor(img, mstype.float32))
        to_image_list([Tensor(img,mstype.float32),Tensor(img,mstype.float32)])
        to_image_list((Tensor(img,mstype.float32),Tensor(img,mstype.float32)))
    
    @pytest.mark.img
    def test_tols_target(self):
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
            print(img)
        to_image_target_list(Tensor([img, img],mstype.float32))
        with pytest.raises(AssertionError):
            to_image_target_list(Tensor(img, mstype.float32))
        to_image_target_list([Tensor(img,mstype.float32),Tensor(img,mstype.float32)])
        to_image_target_list((Tensor(img,mstype.float32),Tensor(img,mstype.float32)))    

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/images.html', '-m=img'])