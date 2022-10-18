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
from src.model_utils.config import config
import pickle as pkl


class TestMask(object):
    @pytest.mark.p
    def test_pooler(self):
        with open('unittest/case/target.pkl', 'rb') as f:
            target = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        part = Pooler(480, 640, Tensor([1,1,1,1,1], mstype.float32))
        out = part(Tensor(img),target)
    


if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/roi_pooler.html', '-m=p'])