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

class TestDataset(object):
    @pytest.mark.d
    def test_normal_input(self):
        dm = DatasetsManager(config=config)
        dm.init_mindrecords()
        dm.init_dataset()

    @pytest.mark.d
    def test_get_data(self):
        dm = DatasetsManager(config=config)
        ds = dm.init_dataset()
        print(ds.batch(1))
    
    @pytest.mark.d
    def test_restore_dataset(self):
        dm = DatasetsManager(config=config)
        with open('datasets/icdar2013/train_images/100.jpg', 'rb') as f:
            img = f.read()
        with open('datasets/icdar2013/train_gts/100.jpg.txt', 'rb') as f:
            gt = f.read()
        image, target = dm.restore_dataset(img, gt)
        print(target)

if __name__ == '__main__':
    pytest.main(['-vv','-s','--html=unittest/results/dataset.html', '-m=d'])