import time
import os
import numpy as np
from numpy import random
import cv2

from src.model_utils.config import config

import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.context import ParallelMode

def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="masktextspotter.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict, masks, masks_shape = create_coco_label(is_training)
    else:
        print("Error unsupported other dataset")
        return

    info_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
        "mask": {"type": "bytes"},
        "mask_shape": {"type": "int32", "shape": [-1]},
    }
    writer.add_schema(info_json, "info_json")

    image_files_num = len(image_files)
    for ind, image_name in enumerate(image_files):
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        mask = masks[image_name]
        mask_shape = masks_shape[image_name]
        row = {"image": img, "annotation": annos, "mask": mask, "mask_shape": mask_shape}
        if (ind + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()


def create_mindrecord_dir(prefix, mindrecord_dir, mindrecord_file):
    if not os.path.isdir(mindrecord_dir):
        os.makedirs(mindrecord_dir)
    if config.dataset == "coco":
        if os.path.isdir(config.coco_root):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("coco", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("coco_root not exits.")
    else:
        if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("other", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("IMAGE_DIR or ANNO_PATH not exits.")
    while not os.path.exists(mindrecord_file+".db"):
        time.sleep(5)