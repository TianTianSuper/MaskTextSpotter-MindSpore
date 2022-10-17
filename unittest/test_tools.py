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
from src.model_utils.tools import SafeConcat
from src.model_utils.config import config
import pickle as pkl


class TestLayer(object):
    @pytest.mark.sc
    def test_sc(self):
        nm = SafeConcat()
        t1 = Tensor([1,2,3,4],mstype.float32)
        t2 = Tensor([2.,3.,4.,5.],mstype.float32)
        assert (nm((t1,t2)) == Tensor([1,2,3,4,2,3,4,5],mstype.float32)).all()   
        # assert nm((np.array([2,3],np.float32), np.array([3,4], np.float32)))

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/tools.html', '-m=sc'])