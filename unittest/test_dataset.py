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
from src.dataset.generator import DatasetsManager
from src.model_utils.config import config

class Testfpn(object):
    @pytest.mark.d
    def test_normal_input(self):
        dm = DatasetsManager(config=config)
        dm.init_mindrecords()
        dm.init_dataset()


if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/dataset.html', '-m=d'])