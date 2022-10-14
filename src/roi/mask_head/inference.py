import mindspore
from mindspore import nn, ops, Tensor
from mindspore.ops import operations as P
from ...model_utils.bounding_box import Boxes


class CharMaskPostProcessor(nn.Cell):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg, masker=None):
        super(CharMaskPostProcessor, self).__init__()
        self.masker = masker
        self.cfg = cfg

    def construct(self, x, char_mask, boxes, seq_outputs=None, seq_scores=None, detailed_seq_scores=None):
        """
        Arguments:
            x (Tensor): the mask logits
            char_mask (Tensor): the char mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        if x is not None:
            mask_prob = P.Sigmoid()(x)
            mask_prob = mask_prob.squeeze(dim=1)[:, None]
            if self.masker:
                mask_prob = self.masker(mask_prob, boxes)
        boxes_per_image = [len(box) for box in boxes]
        if x is not None:
            mask_prob = mask_prob.split(boxes_per_image, dim=0)
        if self.cfg.MODEL.CHAR_MASK_ON:
            char_mask_softmax = P.Softmax(1)(char_mask)
            char_results = {'char_mask': char_mask_softmax.cpu().numpy(), 'boxes': boxes[0].bbox.cpu().numpy(), 'seq_outputs': seq_outputs, 'seq_scores': seq_scores, 'detailed_seq_scores': detailed_seq_scores}
        else:
            char_results = {'char_mask': None, 'boxes': boxes[0].bbox.cpu().numpy(), 'seq_outputs': seq_outputs, 'seq_scores': seq_scores, 'detailed_seq_scores': detailed_seq_scores}
        results = []
        if x is not None:
            for prob, box in zip(mask_prob, boxes):
                bbox = Boxes(box.bbox, box.size, mode="xyxy")
                for field in box.fields():
                    bbox.add_field(field, box.get_field(field))
                bbox.add_field("mask", prob)
                results.append(bbox)
        else:
            for box in boxes:
                bbox = Boxes(box.bbox, box.size, mode="xyxy")
                for field in box.fields():
                    bbox.add_field(field, box.get_field(field))
                results.append(bbox)

        return [results, char_results]