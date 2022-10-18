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
from src.masktextspotter.spn import FpnBlock, SEGHead
from src.masktextspotter.inference import SEGPostHandler
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
        # with open('unittest/case/img.pkl', 'rb') as f:
        #     img = pkl.load(f)
        # img = Tensor(img, mstype.float32)
        # img_2 = img.copy()
        # img_all = mindspore.ops.concat((img,img_2))
        # img_all = img_all.reshape((2, 1,3,480,640))
        # test_part = SEGHead(3, config)
        # output = test_part(img_all)
    
    @pytest.mark.spn
    def test_SegPost(self):
        with open('unittest/case/segmentations.pkl', 'rb') as f:
            seg = pkl.load(f)
        with open('unittest/case/img.pkl', 'rb') as f:
            img = pkl.load(f)
        with open('unittest/case/target.pkl', 'rb') as f:
            target = pkl.load(f)
        # Warning: This part is delayed.
        test_part = SEGPostHandler(config)
        output = test_part(seg[0], img.shape[-2:], target)

if __name__ == '__main__':
    from src.model_utils.config import config
    from mindspore import context
    context.set_context(device_target='CPU',mode=context.PYNATIVE_MODE)

    pytest.main(['-vv','-s','--html=unittest/results/spn.html', '-m=spn'])