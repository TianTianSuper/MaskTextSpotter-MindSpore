import mindspore.numpy as np
from mindspore import ops
from mindspore.ops import composite as C

class BoxCoder(object):
    def __init__(self, weights, clip=None):
        self.weights = weights
        if clip is None:
            clip = np.log(1000.0 / 16)
        self.clip = clip
    
    def encode(self, boxes, proposals):
        remove = 1
        ex_widths = proposals[:, 2] - proposals[:, 0] + remove
        ex_heights = proposals[:, 3] - proposals[:, 1] + remove
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = boxes[:, 2] - boxes[:, 0] + remove
        gt_heights = boxes[:, 3] - boxes[:, 1] + remove
        gt_ctr_x = boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * np.log(gt_widths / ex_widths)
        targets_dh = wh * np.log(gt_heights / ex_heights)

        targets = ops.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, codes, boxes):
        boxes = boxes.astype(codes.type)
        remove = 1
        widths = boxes[:, 2] - boxes[:, 0] + remove
        heights = boxes[:, 3] - boxes[:, 1] + remove
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = codes[:, 0::4] / wx
        dy = codes[:, 1::4] / wy
        dw = codes[:, 2::4] / ww
        dh = codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = C.clip_by_value(dw, max=self.bbox_xform_clip)
        dh = C.clip_by_value(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = np.exp(dw) * widths[:, None]
        pred_h = np.exp(dh) * heights[:, None]

        pred_boxes = np.zeros_like(codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes