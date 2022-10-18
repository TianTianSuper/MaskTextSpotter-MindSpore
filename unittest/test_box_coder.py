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
from src.model_utils.box_coder import BoxCoder
from src.model_utils.config import config
import pickle as pkl


class TestLayer(object):
    @pytest.mark.bc
    def test_bc(self):
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        with open('unittest/case/boxes.pkl', 'rb') as f:
            box_raw = pkl.load(f)

        code = box_coder.encode(Tensor(box_raw), Tensor(box_raw))
        box = box_coder.decode(code, Tensor(box_raw))

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/box_coder.html', '-m=bc'])