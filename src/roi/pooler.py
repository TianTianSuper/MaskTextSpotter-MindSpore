import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.nn import layer as L
from mindspore.common.tensor import Tensor

from .roi_align import ROIAlign
from ..model_utils.tools import SafeConcat

class Level(object):
    def __init__(self, k_low, k_high, scale=224, level=4, e=1e-6):
        super(Level, self).__init__()
        self.k_low = k_low
        self.k_high = k_high
        self.scale = scale
        self.level = level
        self.e = e

        # utils
        self.sqrt = P.Sqrt()
        self.s_concat = SafeConcat()
    
    def __call__(self, boxes_list):
        sc = self.sqrt(self.s_concat([box.area() for box in boxes_list]))
        target = np.floor(self.level + np.log2(sc/self.scale+self.e))
        target = C.clip_by_value(target, self.k_low, self.k_high)
        return target.astype(mstype.int64) - self.k_low
        

class Pooler(object):
    def __init__(self, 
                 out_size_h,
                 out_size_w,
                 scales,
                 sample_num=0,
                 roi_align_mode=1):
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(out_size_h, out_size_w, scale, sample_num))
        self.poolers = nn.CellList(poolers)
        self.out_size_h = out_size_h
        self.out_size_w = out_size_w

        level_min = -np.log2(Tensor(scales[0]))
        level_max = -np.log2(Tensor(scales[-1]))

        self.map_levels = Level(level_min, level_max)

        self.s_concat = P.Concat(0)
        self.concat = P.Concat(1)

    def transfer_to_roi(self, boxes_list):
        concat_boxes = self.concat([box.bbox for box in boxes_list])
        idx = self.s_concat([np.full((len(box), 1), i, dtype=concat_boxes.dtype) for i, box in enumerate(boxes_list)])
        rois = self.concat((idx, concat_boxes))
        return rois

    def __call__(self, input, boxes_list):
        levels_count = len(self.poolers)
        rois = self.transfer_to_roi(boxes_list)
        if levels_count == 1:
            return self.poolers[0](input[0], rois)
        levels = self.map_levels(boxes_list)
        rois_count = len(rois)
        channels_count = input[0].shape[1]
        data_type = input[0].dtype
        res = np.zeros((rois_count, channels_count, self.out_size_h, self.out_size_w), data_type)
        for level, (level_feat, pooler) in enumerate(zip(input, self.poolers)):
            # TODO: whether this operation is necessary
            idx = ops.nonzero(level == levels).squeeze(1)
            rois_of_level = rois[idx]
            res[idx] = pooler(level_feat, rois_of_level)
        
        return res