import mindspore
import mindspore.numpy as np
from mindspore import Tensor, ops, nn
from mindspore.ops import operations as P

from ...model_utils.box_coder import BoxCoder
from ...model_utils.bounding_box import Boxes
from ...model_utils.bbox_ops import boxlist_nms, cat_boxlist

class PostHandler(nn.Cell):
    def __init__(self, thresh=0.05, nms=0.5, detections_of_img=100, box_coder=None, config=None):
        super(PostHandler, self).__init__()
        self.config = config
        self.thresh = thresh
        self.nms = nms
        self.detections_of_img = detections_of_img
        if config.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            if box_coder is None:
                box_coder = BoxCoder(weights=(10., 10., 5., 5.))
            self.box_coder = box_coder
        
        self.top_k = ops.TopK()
        self.softmax = P.Softmax()
        self.concat = P.Concat()

    def combine_boxes(self, boxes, marks, img_shape, mask=None):
        if not self.config.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            scores = scores.reshape(-1)
            boxes.add_field("scores", scores)
            return boxes
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = Boxes(boxes, img_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        if mask is not None:
            boxlist.add_field('masks', mask)
        return boxlist

    def compute_results(self, boxes, classes_count):
        all_boxes = Boxes.bbox.reshape(-1, classes_count * 4)
        marks = Boxes.get_field("scores").reshape(-1, classes_count)

        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = marks > self.thresh
        for j in range(1, classes_count):
            inds = inds_all[:, j].nonzero().squeeze(1)
            marks_j = marks[inds, j]
            boxes_j = all_boxes[inds, j * 4 : (j + 1) * 4]
            boxes_for_class = Boxes(boxes_j, boxes.size, mode="xyxy")
            boxes_for_class.add_field("scores", marks_j)
            boxes_for_class = boxlist_nms(
                boxes_for_class, self.nms, score_field="scores"
            )
            num_labels = len(boxes_for_class)
            boxes_for_class.add_field(
                "labels", np.full((num_labels,), j, dtype=mindspore.int64)
            )
            if self.config.MODEL.SEG.USE_SEG_POLY or self.config.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE or self.config.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
                boxes_for_class.add_field('masks', boxes.get_field('masks'))
            result.append(boxes_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_marks = result.get_field("scores")
            img_thresh, _ = self.top_k(
                cls_marks, number_of_detections - self.detections_per_img + 1
            )
            img_thresh = img_thresh[-1]
            keep = cls_marks >= img_thresh.item()
            keep = ops.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def construct(self, inputs, boxes):
        class_logits, box_regression = inputs
        class_prob = self.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes 
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        if self.config.MODEL.SEG.USE_SEG_POLY or self.config.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE or self.config.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
            masks = [box.get_field('masks') for box in boxes]
        if self.config.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            concat_boxes = self.concat([a.bbox for a in boxes])
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
            proposals = proposals.split(boxes_per_image, dim=0)
        else:
            proposals = boxes
        num_classes = class_prob.shape[1]
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        if self.config.MODEL.SEG.USE_SEG_POLY or self.config.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE or self.config.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
            for prob, boxes_per_img, image_shape, mask in zip(
                class_prob, proposals, image_shapes, masks
            ):
                boxlist = self.combine_boxes(boxes_per_img, prob, image_shape, mask)
                if self.config.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    boxlist = self.compute_results(boxlist, num_classes)
                results.append(boxlist)
        else:
            for prob, boxes_per_img, image_shape in zip(
                class_prob, proposals, image_shapes
            ):
                boxlist = self.combine_boxes(boxes_per_img, prob, image_shape)
                if self.config.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    boxlist = self.compute_results(boxlist, num_classes)
                results.append(boxlist)
        return results
