import mindspore
# import mindspore.numpy as np
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore import context

from .bounding_box import Boxes
from ..masktextspotter.mask import SegmentationMask
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPoint

def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    _box_nms = P.NMSWithMask(nms_thresh)
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    keep = _box_nms(boxlist.pack_field(score_field))
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(
                boxlist1, boxlist2
            )
        )

    # N = len(boxlist1)
    # M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = ops.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = ops.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = C.clip_by_value((rb - lt + TO_REMOVE), clip_value_min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat
    avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return ops.concat(tensors, dim)

def _cat_mask(masks):
    polygons_cat = []
    size = masks[0].size
    for mask in masks:
        polygons = mask.get_polygons()
        polygons_cat.extend(polygons)
    masks_cat = SegmentationMask(polygons_cat, size)
    return masks_cat


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    # if bboxes is None:
    #     return None
    # if bboxes[0] is None:
    #     bboxes = [bboxes[1]
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, Boxes) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = Boxes(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if field == 'masks':
            data = _cat_mask([bbox.get_field(field) for bbox in bboxes])
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def cat_boxlist_gt(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, Boxes) for bbox in bboxes)

    size = bboxes[0].size
    # bboxes[1].set_size(size)
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    if bboxes[0].bbox.sum().item() == 0:
        cat_boxes = Boxes(bboxes[1].bbox, size, mode)
    else:
        cat_boxes = Boxes(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if bboxes[0].bbox.sum().item() == 0:
            if field == 'masks':
                data = _cat_mask([bbox.get_field(field) for bbox in bboxes[1:]])
            else:
                data = _cat([bbox.get_field(field) for bbox in bboxes[1:]], dim=0)
        else:
            if field == 'masks':
                data = _cat_mask([bbox.get_field(field) for bbox in bboxes])
            else:
                data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
