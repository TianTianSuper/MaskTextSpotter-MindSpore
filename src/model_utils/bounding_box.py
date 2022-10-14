import mindspore
import mindspore.numpy as np
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore import context

class Boxes(object):
    def __init__(self, bbox, image_size, mode="xyxy", use_char_ann=True, is_fake=False):
        bbox = Tensor(bbox, mindspore.float32)
        if bbox.ndim != 2:
            raise ValueError("bbox should have 2 dimensions, but got {}".format(bbox.ndim))
        if bbox.shape[-1] != 4:
            raise ValueError("size of last dimension of bbox should be 4, but got {}".format(bbox.shape[-1]))
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'.")
        
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}
        self.use_char_ann = use_char_ann

        # utils
        self.isinstance = P.IsInstance()
        self.split_xyxy = P.Split(-1, self.bbox.shape[-1])
        self.concat = P.Concat(-1)
        self.field_concat = P.Concat(1)

        # transpose
        self.flip_left_right = 0
        self.flip_top_bottom = 1
    
    def add_field(self, field, data):
        self.extra_fields[field] = data

    def get_field(self, field):
        return self.extra_fields[field]
    
    def has_field(self, field):
        return field in self.extra_fields
    
    def fields(self):
        return list(self.extra_fields.keys())
    
    def _copy_extra_fields(self, bbox):
        for key, value in bbox.extra_fields.items():
            self.extra_fields[key] = value
    
    def pack_field(self, field):
        if not self.has_field(field):
            raise NameError("Field '{}' not found".format(field))
        return self.field_concat((self.bbox, Tensor(self.extra_fields[field]).expand_dims(1)))

    def _split_to_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.split_xyxy(self.bbox)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            remove = 1
            xmin, ymin, w, h = self.split_xyxy(self.bbox)
            return (
                xmin,
                ymin,
                xmin + C.clip_by_value((w - remove), clip_value_min=0),
                ymin + C.clip_by_value((h - remove), clip_value_min=0)
            )
        else:
            raise RuntimeError("Error in this condition")
    
    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_to_xyxy()
        if mode == "xyxy":
            bbox = self.concat((xmin, ymin, xmax, ymax))
            bbox = Boxes(bbox, self.size, mode=mode, use_char_ann=self.use_char_ann)
        else:
            remove = 1
            bbox = self.concat((xmin, ymin, xmax - xmin + remove, ymax - ymin + remove))
            bbox = Boxes(bbox, self.size, mode=mode, use_char_ann=self.use_char_ann)
        bbox._copy_extra_fields(self)
        return bbox

    def set_size(self, size):
        self.size = size
        bbox = Boxes(self.bbox, size, mode=self.mode, use_char_ann=self.use_char_ann)
        for key, value in self.extra_fields.items():
            if not self.isinstance(value, mstype.tensor):
                value = value.set_size(size)
            bbox.add_field(key, value)
        return bbox.convert(self.mode)
    
    def resize(self, size, *args, **kwargs):
        rates = tuple(float(i) / float(j) for i, j in zip(size, self.size))
        if rates[0] == rates[1]:
            rate = rates[0]
            scaled_box = self.bbox * rate
            bbox = Boxes(scaled_box, size, mode=self.mode, use_char_ann=self.use_char_ann)
            for key, value in self.extra_fields.items():
                if not self.isinstance(value, mstype.tensor):
                    value = value.resize(size, *args, **kwargs)
                bbox.add_field(key, value)
            return bbox
        else:
            rate_width, rate_height = rates
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            scaled_xmin = xmin * rate_width
            scaled_xmax = xmax * rate_width
            scaled_ymin = ymin * rate_height
            scaled_ymax = ymax * rate_height
            scaled_box = self.concat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax))
            bbox = Boxes(scaled_box, size, mode="xyxy", use_char_ann=self.use_char_ann)
            for key, value in self.extra_fields.items():
                if not self.isinstance(value, mstype.tensor):
                    value = value.resize(size, *args, **kwargs)
                bbox.add_field(key, value)
            return bbox.convert(self.mode)
    
    def poly_to_box(self, poly):
        xmin = min(poly[0::2])
        xmax = max(poly[0::2])
        ymin = min(poly[1::2])
        ymax = max(poly[1::2])
        return [xmin, ymin, xmax, ymax]
    
    def rotate(self, angle, r_c, start_h, start_w):
        masks = self.extra_fields["masks"]
        masks = masks.rotate(angle, r_c, start_h, start_w)
        polys = masks.polygons
        boxes = []
        for poly in polys:
            box = self.poly_to_box(poly.polygons[0].numpy())
            boxes.append(box)
        self.size = (r_c[0] * 2, r_c[1] * 2)
        bbox = Boxes(boxes, self.size, mode="xyxy", use_char_ann=self.use_char_ann)
        for key, value in self.extra_fields.items():
            if key == "masks":
                value = masks
            else:
                if self.use_char_ann:
                    if not self.isinstance(value, mstype.tensor):
                        value = value.rotate(angle, r_c, start_h, start_w)
                else:
                    if not self.isinstance(value, mstype.tensor) and key != "char_masks":
                        value = value.rotate(angle, r_c, start_h, start_w)
            bbox.add_field(key, value)
        return bbox.convert(self.mode)
    
    def transpose(self, method):
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_to_xyxy()
        if method == self.flip_left_right:
            remove = 1
            transposed_xmin = image_width - xmax - remove
            transposed_xmax = image_width - xmin - remove
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == self.flip_top_bottom:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = self.concat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax)
        )
        bbox = Boxes(
            transposed_boxes, self.size, mode="xyxy", use_char_ann=self.use_char_ann
        )
        for key, value in self.extra_fields.items():
            if not self.isinstance(value, mstype.tensor):
                value = value.transpose(method)
            bbox.add_field(key, value)
        return bbox.convert(self.mode)

    def crop(self, box):
        xmin, ymin, xmax, ymax = self._split_to_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = C.clip_by_value((xmin - box[0]), 0, w)
        cropped_ymin = C.clip_by_value((ymin - box[1]), 0, h)
        cropped_xmax = C.clip_by_value((xmax - box[0]), 0, w)
        cropped_ymax = C.clip_by_value((ymax - box[1]), 0, h)

        keep_ind = None
        not_empty = np.where(
            (cropped_xmin != cropped_xmax) & (cropped_ymin != cropped_ymax)
        )[0]
        if len(not_empty) > 0:
            keep_ind = not_empty
        cropped_box = self.concat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax)
        )
        cropped_box = cropped_box[not_empty]
        bbox = Boxes(cropped_box, (w, h), mode="xyxy", use_char_ann=self.use_char_ann)
        for key, value in self.extra_fields.items():
            if self.use_char_ann:
                if not self.isinstance(value, mstype.tensor):
                    value = value.crop(box, keep_ind)
            else:
                if not self.isinstance(value, mstype.tensor) and key != "char_masks":
                    value = value.crop(box, keep_ind)
            bbox.add_field(key, value)
        return bbox.convert(self.mode)

    def __len__(self):
        return self.bbox.shape[0]
    
    def __getitem__(self, item):
        bbox = Boxes(self.bbox[item], self.size, self.mode, self.use_char_ann)
        for key, value in self.extra_fields.items():
            bbox.add_field(key, value[item])
        return bbox

    def clip_to_image(self, remove_empty=True):
        remove = 1
        self.bbox[:, 0] = C.clip_by_value(self.bbox[:, 0], 0, self.size[0] - remove)
        self.bbox[:, 1] = C.clip_by_value(self.bbox[:, 1], 0, self.size[1] - remove)
        self.bbox[:, 2] = C.clip_by_value(self.bbox[:, 2], 0, self.size[0] - remove)
        self.bbox[:, 3] = C.clip_by_value(self.bbox[:, 3], 0, self.size[1] - remove)

        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        remove = 1
        box = self.bbox
        area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)
        return area

    def copy_with_fields(self, fields):
        bbox = Boxes(self.bbox, self.size, self.mode, self.use_char_ann)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

if __name__ == "__main__":
    context.set_context(device_target='GPU',mode=context.GRAPH_MODE,enable_graph_kernel=True)
    bbox = Boxes([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
