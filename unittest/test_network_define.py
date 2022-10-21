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
from src.network_define import LossCallBack
from src.model_utils.config import config
import pickle as pkl
from easydict import EasyDict
class ExampleContext:
    def __init__(self, original_args) -> None:
        self.oa = original_args
    def original_args(self):
        return self.oa
class OA:
    def __init__(self) -> None:
        self.cur_step_num = 0
        self.batch_num = 32
        self.cur_epoch_num = 0
        self.net_outputs = Tensor([123,],mstype.float32)
    def get(self):
        return self.net_outputs

class TestCallback(object):
    @pytest.mark.cb
    def test_init(self):
        loss_cb = LossCallBack()

    @pytest.mark.cb
    def test_step_end(self):
        loss_cb = LossCallBack()

        oa = OA()
        test_context = ExampleContext(oa)
        loss_cb.step_end(test_context)

'''
output to ./loss_0.log successfully
0 epoch: 0 step: 32 total_loss: 123.00000
'''

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/network_define.html', '-m=cb'])