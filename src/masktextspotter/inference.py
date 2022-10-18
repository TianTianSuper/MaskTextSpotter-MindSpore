import mindspore
from mindspore import Tensor
from mindspore import numpy as np
from mindspore import nn, ops
from mindspore.ops import operations as P
from ..model_utils.bounding_box import Boxes
from .mask import SegmentationMask
from ..model_utils.bbox_ops import cat_boxlist, cat_boxlist_gt

import cv2
import random
import pyclipper
from shapely.geometry import Polygon

class SEGPostHandler(nn.Cell):
    def __init__(self, config, train_status=True):
        super(SEGPostHandler, self).__init__()
        self.top_n = config.top_n_train
        if not train_status:
            self.top_n = config.top_n_test

        self.binary_thresh = config.binary_thresh
        self.box_thresh = config.box_thresh
        self.min_size = config.min_size
        self.config = config

        # utils
        self.concat = P.Concat(1)

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[Boxes]
            targets: list[Boxes]
        """
        gt_boxes = [target.copy_with_fields(['masks']) for target in targets]
        proposals = [
            cat_boxlist_gt([proposal, gt_box])
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def aug_tensor_proposals(self, boxes):
        # boxes: N * 4
        boxes = boxes.astype(mindspore.float32)
        N = boxes.shape[0]
        aug_boxes = np.zeros((4, N, 4))
        aug_boxes[0, :, :] = boxes.copy()
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        width = xmax - xmin
        height = ymax - ymin
        for i in range(3):
            choice = random.random()
            if choice < 0.5:
                ratio = (np.randn((N,)) * 3 + 1) / 2.
                height = height * ratio
                ratio = (np.randn((N,)) * 3 + 1) / 2.
                width = width * ratio
            else:
                move_x = width * (np.randn((N,)) * 4 - 2)
                move_y = height * (np.randn((N,)) * 4 - 2)
                x_center += move_x
                y_center += move_y
            boxes[:, 0] = x_center - width / 2
            boxes[:, 2] = x_center + width / 2
            boxes[:, 1] = y_center - height / 2
            boxes[:, 3] = y_center + height / 2
            aug_boxes[i+1, :, :] = boxes.copy()
        return aug_boxes.reshape((-1, 4))

    def forward_for_single_feature_map(self, pred, image_shapes):
        """
        Arguments:
            pred: tensor of size N, 1, H, W
        """
        bitmap = self.binarize(pred)
        N, height, width = pred.shape[0], pred.shape[2], pred.shape[3]
        bitmap_numpy = bitmap.asnumpy() 
        pred_map_numpy = pred.asnumpy()
        boxes_batch = []
        rotated_boxes_batch = []
        polygons_batch = []
        scores_batch = []
        for batch_index in range(N):
            image_shape = image_shapes[batch_index]
            boxes, scores, rotated_boxes, polygons = self.boxes_from_bitmap(
                pred_map_numpy[batch_index],
                bitmap_numpy[batch_index], width, height)
            if self.training and self.config.MODEL.SEG.AUG_PROPOSALS:
                boxes = self.aug_tensor_proposals(boxes)
            if boxes.shape[0] > self.top_n:
                boxes = boxes[:self.top_n, :]
            boxlist = Boxes(boxes, (image_shape[1], image_shape[0]), mode="xyxy")
            masks = SegmentationMask(polygons, (image_shape[1], image_shape[0]))
            boxlist.add_field('masks', masks)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxes_batch.append(boxlist)
            rotated_boxes_batch.append(rotated_boxes)
            polygons_batch.append(polygons)
            scores_batch.append(scores)
        return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch

    def binarize(self, pred):
        return ops.less(self.binary_thresh, Tensor(pred))

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """
        # assert _bitmap.size(0) == 1
        # bitmap = _bitmap[0]  # The first channel
        pred = pred[0]
        height, width = bitmap.shape[1], bitmap.shape[2]
        boxes = []
        scores = []
        rotated_boxes = []
        polygons = []
        contours_all = []
        for i in range(bitmap.shape[0]):
            try:
                _, contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            except BaseException:
                contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            contours_all.extend(contours)
        for contour in contours_all:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape((-1, 2))
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points)
            if not self.training and self.box_thresh > score:
                continue
            if polygon.shape[0] > 2:
                polygon = self.unclip(polygon, expand_ratio=self.config.expand_rate)
                if len(polygon) > 1:
                    continue
            else:
                continue
            # polygon = polygon.reshape(-1, 2)
            polygon = polygon.reshape(-1)
            box = self.unclip(points, expand_ratio=self.config.expand_rate).reshape(-1, 2)
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            min_x, min_y = min(box[:, 0]), min(box[:, 1])
            max_x, max_y = max(box[:, 0]), max(box[:, 1])
            horizontal_box = Tensor(np.array([min_x, min_y, max_x, max_y]))
            boxes.append(horizontal_box)
            scores.append(score)
            rotated_box, _ = self.get_mini_boxes(box.reshape(-1, 1, 2))
            rotated_box = np.array(rotated_box)
            rotated_boxes.append(rotated_box)
            polygons.append([polygon])
        if len(boxes) == 0:
            boxes = [Tensor(np.array([0, 0, 0, 0]))]
            scores = [0.]

        boxes = ops.stack(boxes)
        scores = Tensor(np.array(scores))
        return boxes, scores, rotated_boxes, polygons

    def aug_proposals(self, box):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        x_center = int((xmin + xmax) / 2.)
        y_center = int((ymin + ymax) / 2.)
        width = xmax - xmin
        height = ymax - ymin
        choice = random.random()
        if choice < 0.5:
            # shrink or expand
            ratio = (random.random() * 3 + 1) / 2.
            height = height * ratio
            ratio = (random.random() * 3 + 1) / 2.
            width = width * ratio
        else:
            move_x = width * (random.random() * 4 - 2)
            move_y = height * (random.random() * 4 - 2)
            x_center += move_x
            y_center += move_y
        xmin = int(x_center - width / 2)
        xmax = int(x_center + width / 2)
        ymin = int(y_center - height / 2)
        ymax = int(y_center + height / 2)
        return [xmin, ymin, xmax, ymax]

    def unclip(self, box, expand_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * expand_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score(self, bitmap, box):
        """
        naive version of box score computation,
        only for helping principle understand.
        """
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap, mask)[0]

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
    
    def construct(self, seg_output, image_shapes, targets=None):
        sampled_boxes = []
        boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch \
             = self.forward_for_single_feature_map(seg_output, image_shapes)
        if not self.training:
            return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch
        sampled_boxes.append(boxes_batch)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists
