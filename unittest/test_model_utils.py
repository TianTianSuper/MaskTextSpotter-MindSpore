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
from src.model_utils import *
from src.model_utils.config import config

class TestSpn(object):
    # @pytest.mark.mu_remove
    # def test_assigner(self):
    #     # assigner should be remove
    #     test_part = assigner.ElementsAssigner(1,0)
    #     input_data = Tensor(np.ones((1, 256, 16, 16)), mstype.float32)
    #     output = test_part(input_data)
    @pytest.mark.mu
    def test_bounding_box(self):
        bbox = bounding_box.Boxes([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
        s_bbox = bbox.resize((5, 5))
        s_bbox.add_field("test", [[1., 1., 12., 22.], [3., 2., 1., 9.]])
        assert s_bbox.get_field("test") == [[1., 1., 12., 22.], [3., 2., 1., 9.]]
        assert s_bbox.has_field("test") == True
        assert s_bbox.has_field("testtt") == False
        assert type(s_bbox.fields()) == list
        # assert s_bbox.pack_field("test")
        # assert type(s_bbox.pack_field("test")) == Tensor
        with pytest.raises(NameError):
            s_bbox.pack_field("testtt")
        with pytest.raises(ValueError):
            s_bbox.convert("xxxx")
        assert s_bbox.convert("xyxy") == s_bbox
        assert s_bbox.convert("xywh") != s_bbox
        # assert s_bbox.set_size((10,10))

        t_bbox = bbox.transpose(0)
        print(t_bbox)
        print(t_bbox.bbox)


    # @pytest.mark.spn
    # def test_SEGHead(self):
    #     # Warning: this case will raise RuntimeError In PyTorch
    #     # due to the shape of p5,p4 resulting to concat failed.
    #     # The same reason to ValueError in mindspore
    #     with pytest.raises(ValueError):
    #         test_part = SEGHead(256, config)
    #         input_data = Tensor(np.ones(327680).reshape((1, 5, 256, 16, 16)).T, mstype.float32)
    #         output = test_part(input_data)
    
    # @pytest.mark.spn
    # def test_SegPost(self):
    #     # Warning: This part is delayed.
    #     test_part = SEGPostHandler(config)
    #     input_data = Tensor(np.ones(327680).reshape((1, 5, 256, 16, 16)).T, mstype.float32)
    #     img_size = Tensor((16,16), dtype=mstype.float32)
    #     output = test_part(input_data, img_size, input_data)

if __name__ == '__main__':
    from src.model_utils.config import config
    from mindspore import context
    context.set_context(device_target='GPU',mode=context.PYNATIVE_MODE)

    pytest.main(['-vv','-s','--html=unittest/results/model_utils.html', '-m=mu'])