import time
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from numpy import random
import cv2
import logging

from src.model_utils.config import config

import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.context import ParallelMode

def create_mindrecords(config, is_training=True, prefix="masktextspotter.mindrecord", file_num=8):
    """Create MindRecord file."""
    datasets = config.datasets_name
    mindrecord_dir = config.mindrecord_dir
    if not os.path.exists(mindrecord_dir):
        os.mkdir(mindrecord_dir)
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    info_json = {
        "image": {"type": "bytes"},
        "ground_trues": {"type": "bytes"},
    }
    writer.add_schema(info_json, "info_json")

    for dataset in datasets:
        if dataset not in ("icdar2013", "icdar2015", "synthtext", "total_text", "scut-eng-char"):
            logging.warning(
                "Dataset '{}' is not in recommended datasets list: \
                'icdar2013', 'icdar2015', 'synthtext', 'total_text', 'scut-eng-char'.".format(dataset))
        if is_training:
            image_files_path = os.path.join(config.datasets_root, dataset+'/train_images')
            gts_files_path = os.path.join(config.datasets_root, dataset+'/train_gts')
        else:
            image_files_path = os.path.join(config.datasets_root, dataset+'/test_images')
            gts_files_path = os.path.join(config.datasets_root, dataset+'/test_gts')
        image_files = os.listdir(image_files_path)

        print("Transferring '{}':".format(dataset))
        image_files_num = len(image_files)
        for i, image_name in enumerate(image_files):
            with open(os.path.join(image_files_path, image_name), 'rb') as f:
                img = f.read()
            if not os.path.exists(os.path.join(gts_files_path, image_name+'.txt')):
                continue
            with open(os.path.join(gts_files_path, image_name+'.txt'), 'rb') as f:
                gts = f.read()
            raw = {"image": img, "ground_trues": gts}
            
            if (i + 1) % 10 == 0:
                print("writing {}/{} into mindrecord".format(i + 1, image_files_num))
            writer.write_raw_data([raw])

    writer.commit()


if __name__ == '__main__':
    create_mindrecords(config)