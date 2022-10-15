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
from src.masktextspotter.spn import FpnBlock, SEGHead
from src.model_utils.config import config

class TestSpn(object):
    @pytest.mark.spn
    def test_fpn_block(self):
        test_part = FpnBlock(256, 512)
        input_data = Tensor(np.ones((1, 256, 16, 16)), mstype.float32)
        output = test_part(input_data)

    @pytest.mark.spn
    def test_SEGHead(self):
        # Warning: this case will raise RuntimeError In PyTorch
        # due to the shape of p5,p4 resulting to concat failed.
        # The same reason to ValueError in mindspore
        with pytest.raises(ValueError):
            test_part = SEGHead(256, config)
            input_data = Tensor(np.ones(327680).reshape((1, 5, 256, 16, 16)).T, mstype.float32)
            output = test_part(input_data)

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/spn.html', '-m=spn'])