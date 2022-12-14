import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np

import mindspore
import mindspore.numpy as msnp
from mindspore import dataset
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter, FileReader
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.context import ParallelMode

from src.model_utils.bounding_box import Boxes
from src.model_utils.config import config
from src.masktextspotter.mask import SegmentationMask, SegmentationCharMask

import logging



class DatasetsManager:

    def __init__(self, config, training_status=True, prefix="masktextspotter.mindrecord", file_num=1):
        self.datasets_name = config.datasets_name
        self.datasets_count = len(self.datasets_name)
        self.mindrecord_dir = config.mindrecord_dir
        self.train_status = training_status
        self.prefix = prefix
        self.record_num = file_num
        self.workers = config.data_workers
        self.batch_size = config.batch_size
        self.config = config
        self.ignore_difficult = False
        self.use_charann = True
        self.char_classes = "_0123456789abcdefghijklmnopqrstuvwxyz"

    def gt2boxes(self, gt):
        parts = gt.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            parts[0] = parts[0][3:]
        if "\ufeff" in parts[0]:
            parts[0] = parts[0].replace("\ufeff", "")
        x1 = np.array([int(float(x)) for x in parts[::9]])
        y1 = np.array([int(float(x)) for x in parts[1::9]])
        x2 = np.array([int(float(x)) for x in parts[2::9]])
        y2 = np.array([int(float(x)) for x in parts[3::9]])
        x3 = np.array([int(float(x)) for x in parts[4::9]])
        y3 = np.array([int(float(x)) for x in parts[5::9]])
        x4 = np.array([int(float(x)) for x in parts[6::9]])
        y4 = np.array([int(float(x)) for x in parts[7::9]])
        strs = parts[8::9]
        loc = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
        return strs, loc

    def init_mindrecords(self):
        if not os.path.exists(self.mindrecord_dir):
            os.mkdir(self.mindrecord_dir)        
        mindrecord_path = os.path.join(self.mindrecord_dir, self.prefix)
        writer = FileWriter(mindrecord_path, self.record_num)
        info_json = {
            "image": {"type": "bytes"},
            "ground_trues": {"type": "bytes"},
        }
        writer.add_schema(info_json, "info_json")

        for dataset in self.datasets_name:
            if dataset not in ("icdar2013", "icdar2015", "synthtext", "total_text", "scut-eng-char"):
                logging.warning(
                    "Dataset '{}' is not in recommended datasets list: \
                    'icdar2013', 'icdar2015', 'synthtext', 'total_text', 'scut-eng-char'.".format(dataset))
            if self.train_status:
                image_files_path = os.path.join(self.config.datasets_root, dataset+'/train_images')
                gts_files_path = os.path.join(self.config.datasets_root, dataset+'/train_gts')
            else:
                image_files_path = os.path.join(self.config.datasets_root, dataset+'/test_images')
                gts_files_path = os.path.join(self.config.datasets_root, dataset+'/test_gts')
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
        return True
    
    def transfer(self, image, ground_trues):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gts = ground_trues.decode('utf-8') \
                          .replace('\r','\n').replace('\n\n', '\n').replace('\n',' ') \
                          .rstrip().split(' ')
        words, boxes, charsboxes, segmentations, labels = [], [], [], [], []
        for gt in gts:
            charbbs = []
            strs, loc = self.gt2boxes(gt)
            word = strs[0]
            if word == "###":
                if self.ignore_difficult:
                    rect = list(loc[0])
                    min_x = min(rect[::2]) - 1
                    min_y = min(rect[1::2]) - 1
                    max_x = max(rect[::2]) - 1
                    max_y = max(rect[1::2]) - 1
                    box = [min_x, min_y, max_x, max_y]
                    segmentations.append([loc[0, :]])
                    tindex = len(boxes)
                    boxes.append(box)
                    words.append(word)
                    labels.append(-1)
                    charbbs = np.zeros((10,), dtype=np.float32)
                    if loc.shape[0] > 1:
                        for i in range(1, loc.shape[0]):
                            charbb[9] = tindex
                            charbbs.append(charbb.copy())
                        charsboxes.append(charbbs)
                else:
                    continue
            else:
                rect = list(loc[0])
                min_x = min(rect[::2]) - 1
                min_y = min(rect[1::2]) - 1
                max_x = max(rect[::2]) - 1
                max_y = max(rect[1::2]) - 1
                box = [min_x, min_y, max_x, max_y]
                segmentations.append([loc[0, :]])
                tindex = len(boxes)
                boxes.append(box)
                words.append(word)
                labels.append(1)
                c_class = self.char2num(strs[1:])
                charbb = np.zeros((10,), dtype=np.float32)
                if loc.shape[0] > 1:
                    for i in range(1, loc.shape[0]):
                        charbb[:8] = loc[i, :]
                        charbb[8] = c_class[i - 1]
                        charbb[9] = tindex
                        charbbs.append(charbb.copy())
                    charsboxes.append(charbbs)
        num_boxes = len(boxes)
        if len(boxes) > 0:
            keep_boxes = np.zeros((num_boxes, 5))
            keep_boxes[:, :4] = np.array(boxes)
            keep_boxes[:, 4] = range(
                num_boxes # ???box???????????????????????????
            )
            if not self.use_charann:
                charbbs = np.zeros((10,), dtype=np.float32)
                if len(charsboxes) == 0:
                    for _ in range(len(words)):
                        charsboxes.append([charbbs])
                return (
                    image_bgr,
                    words,
                    np.array(keep_boxes),
                    charsboxes,
                    segmentations,
                    labels
                )
            else:
                return image_bgr, words, np.array(keep_boxes), charsboxes, segmentations, labels
 

        else:
            words.append("")
            charbbs = np.zeros((10,), dtype=np.float32)
            return (
                image_bgr,
                words,
                np.zeros((1, 5), dtype=np.float32),
                [[charbbs]],
                [[np.zeros((8,), dtype=np.float32)]],
                [1]
            )

    def restore_dataset(self, image, ground_trues):
        image, words, boxes, charsbbs, segmentations, labels = self.transfer(image, ground_trues)
        image_shape = image.shape[-2:]
        target = Boxes(
            boxes[:, :4], image_shape, mode="xyxy", use_char_ann=self.use_charann
        )
        if self.ignore_difficult:
            labels = msnp.from_numpy(np.array(labels))
        else:
            labels = msnp.ones(len(boxes))
        target.add_field("labels", labels)
        masks = SegmentationMask(segmentations, image_shape)
        target.add_field("masks", masks)
        if words[0] == "":
            use_char_ann = False
        else:
            use_char_ann = True
        if not self.use_charann:
            use_char_ann = False
        char_masks = SegmentationCharMask(
            charsbbs, words=words, use_char_ann=use_char_ann, size=image_shape, char_num_classes=len(self.char_classes)
        )
        target.add_field("char_masks", char_masks)
        return image, target
        
    def char2num(self, chars):
        ## chars ['h', 'e', 'l', 'l', 'o']
        nums = [self.char_classes.index(c.lower()) for c in chars]
        return nums

    def init_dataset(self):
        cv2.setNumThreads(0)
        dataset.config.set_prefetch_size(8)
        load_dir = os.path.join(self.mindrecord_dir, "masktextspotter.mindrecord")
        datacomb = dataset.MindDataset(load_dir)
        decode = C.Decode()
        datacomb = datacomb.map(decode, input_columns=["image"])
        compose_map_func = (lambda image, ground_trues:
                            self.restore_dataset(image, ground_trues))
        
        if self.train_status:
            datacomb = datacomb.map(
                    operations=compose_map_func,
                    input_columns=["image", "ground_trues"],
                    output_columns=["image", "target"],
                    column_order=["image", "target"],
                    python_multiprocessing=False,
                    num_parallel_workers=self.workers)
            datacomb = datacomb.batch(self.batch_size, drop_remainder=True)
        else:
            datacomb = datacomb.map(
                    operations=compose_map_func,
                    input_columns=["image", "ground_trues"],
                    output_columns=["image", "target"],
                    column_order=["image", "target"],
                    python_multiprocessing=False,
                    num_parallel_workers=self.workers)
            datacomb = datacomb.batch(self.batch_size, drop_remainder=True)
        return datacomb


class NormalManager:
    def __init__(self, config, training_status=True):
        self.datasets_name = config.datasets_name
        self.datasets_count = len(self.datasets_name)
        self.train_status = training_status
        self.workers = config.data_workers
        self.batch_size = config.batch_size
        self.config = config
        self.ignore_difficult = False
        self.use_charann = True
        self.char_classes = "_0123456789abcdefghijklmnopqrstuvwxyz"
        self.size = 0

    def char2num(self, chars):
        ## chars ['h', 'e', 'l', 'l', 'o']
        nums = [self.char_classes.index(c.lower()) for c in chars]
        return nums

    def gt2boxes(self, gt):
        parts = gt.strip().split(",")
        if "\xef\xbb\xbf" in parts[0]:
            parts[0] = parts[0][3:]
        if "\ufeff" in parts[0]:
            parts[0] = parts[0].replace("\ufeff", "")
        x1 = np.array([int(float(x)) for x in parts[::9]])
        y1 = np.array([int(float(x)) for x in parts[1::9]])
        x2 = np.array([int(float(x)) for x in parts[2::9]])
        y2 = np.array([int(float(x)) for x in parts[3::9]])
        x3 = np.array([int(float(x)) for x in parts[4::9]])
        y3 = np.array([int(float(x)) for x in parts[5::9]])
        x4 = np.array([int(float(x)) for x in parts[6::9]])
        y4 = np.array([int(float(x)) for x in parts[7::9]])
        strs = parts[8::9]
        loc = np.vstack((x1, y1, x2, y2, x3, y3, x4, y4)).transpose()
        return strs, loc


    def transfer(self, image, ground_trues):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image_bgr = image.copy()
        image_bgr[:, :, 0] = image[:, :, 2]
        image_bgr[:, :, 1] = image[:, :, 1]
        image_bgr[:, :, 2] = image[:, :, 0]
        image_shape = image_bgr.shape[:2]
        gts = ground_trues.decode('utf-8') \
                          .replace('\r','\n').replace('\n\n', '\n').replace('\n',' ') \
                          .rstrip().split(' ')
        words, boxes, charsboxes, segmentations, labels = [], [], [], [], []
        for gt in gts:
            charbbs = []
            strs, loc = self.gt2boxes(gt)
            word = strs[0]
            if word == "###":
                if self.ignore_difficult:
                    rect = list(loc[0])
                    min_x = min(rect[::2]) - 1
                    min_y = min(rect[1::2]) - 1
                    max_x = max(rect[::2]) - 1
                    max_y = max(rect[1::2]) - 1
                    box = [min_x, min_y, max_x, max_y]
                    segmentations.append([loc[0, :]])
                    tindex = len(boxes)
                    boxes.append(box)
                    words.append(word)
                    labels.append(-1)
                    charbbs = np.zeros((10,), dtype=np.float32)
                    if loc.shape[0] > 1:
                        for i in range(1, loc.shape[0]):
                            charbb[9] = tindex
                            charbbs.append(charbb.copy())
                        charsboxes.append(charbbs)
                else:
                    continue
            else:
                rect = list(loc[0])
                min_x = min(rect[::2]) - 1
                min_y = min(rect[1::2]) - 1
                max_x = max(rect[::2]) - 1
                max_y = max(rect[1::2]) - 1
                box = [min_x, min_y, max_x, max_y]
                segmentations.append([loc[0, :]])
                tindex = len(boxes)
                boxes.append(box)
                words.append(word)
                labels.append(1)
                c_class = self.char2num(strs[1:])
                charbb = np.zeros((10,), dtype=np.float32)
                if loc.shape[0] > 1:
                    for i in range(1, loc.shape[0]):
                        charbb[:8] = loc[i, :]
                        charbb[8] = c_class[i - 1]
                        charbb[9] = tindex
                        charbbs.append(charbb.copy())
                    charsboxes.append(charbbs)
        num_boxes = len(boxes)
        if len(boxes) > 0:
            keep_boxes = np.zeros((num_boxes, 5))
            keep_boxes[:, :4] = np.array(boxes)
            keep_boxes[:, 4] = range(
                num_boxes # ???box???????????????????????????
            )
            if not self.use_charann:
                charbbs = np.zeros((10,), dtype=np.float32)
                if len(charsboxes) == 0:
                    for _ in range(len(words)):
                        charsboxes.append([charbbs])
                return (
                    image_bgr,
                    words,
                    np.array(keep_boxes),
                    charsboxes,
                    segmentations,
                    labels
                )
            else:
                return image_bgr, words, np.array(keep_boxes), charsboxes, segmentations, labels
        else:
            words.append("")
            charbbs = np.zeros((10,), dtype=np.float32)
            return (
                image_bgr,
                words,
                np.zeros((1, 5), dtype=np.float32),
                [[charbbs]],
                [[np.zeros((8,), dtype=np.float32)]],
                [1]
            )
    
    def generate(self):
        images_ls = []
        targets_ls = []
        for ds in self.datasets_name:
            if ds not in ("icdar2013", "icdar2015", "synthtext", "total_text", "scut-eng-char"):
                logging.warning(
                    "Dataset '{}' is not in recommended datasets list: \
                    'icdar2013', 'icdar2015', 'synthtext', 'total_text', 'scut-eng-char'.".format(ds))
            if self.train_status:
                image_files_path = os.path.join(self.config.datasets_root, ds+'/train_images')
                gts_files_path = os.path.join(self.config.datasets_root, ds+'/train_gts')
            else:
                image_files_path = os.path.join(self.config.datasets_root, ds+'/test_images')
                gts_files_path = os.path.join(self.config.datasets_root, ds+'/test_gts')
            image_files = os.listdir(image_files_path)

            for i, image_name in enumerate(image_files):
                with open(os.path.join(image_files_path, image_name), 'rb') as f:
                    img = f.read()
                if not os.path.exists(os.path.join(gts_files_path, image_name+'.txt')):
                    continue
                with open(os.path.join(gts_files_path, image_name+'.txt'), 'rb') as f:
                    gts = f.read()
                
                image, words, boxes, charsbbs, segmentations, labels = self.transfer(img, gts)
                image_shape = image.shape[-2:]
                target = Boxes(
                    boxes[:, :4], image_shape, mode="xyxy", use_char_ann=self.use_charann
                )
                if self.ignore_difficult:
                    labels = msnp.from_numpy(np.array(labels))
                else:
                    labels = msnp.ones(len(boxes))
                target.add_field("labels", labels)
                masks = SegmentationMask(segmentations, image_shape)
                target.add_field("masks", masks)
                if words[0] == "":
                    use_char_ann = False
                else:
                    use_char_ann = True
                if not self.use_charann:
                    use_char_ann = False
                char_masks = SegmentationCharMask(
                    charsbbs, words=words, use_char_ann=use_char_ann, size=image_shape, char_num_classes=len(self.char_classes)
                )
                target.add_field("char_masks", char_masks)

                images_ls.append(image)
                targets_ls.append(target)
        self.size = len(images_ls)
        return images_ls, targets_ls
    
    def generate_single(self, image_files_path, gts_files_path, name):
        with open(os.path.join(image_files_path, name), 'rb') as f:
            img = f.read()
        with open(os.path.join(gts_files_path, name+'.txt'), 'rb') as f:
            gts = f.read()
        
        image, words, boxes, charsbbs, segmentations, labels = self.transfer(img, gts)
        image_shape = image.shape[-2:]
        target = Boxes(
            boxes[:, :4], image_shape, mode="xyxy", use_char_ann=self.use_charann
        )
        if self.ignore_difficult:
            labels = msnp.from_numpy(np.array(labels))
        else:
            labels = msnp.ones(len(boxes))
        target.add_field("labels", labels)
        masks = SegmentationMask(segmentations, image_shape)
        target.add_field("masks", masks)
        if words[0] == "":
            use_char_ann = False
        else:
            use_char_ann = True
        if not self.use_charann:
            use_char_ann = False
        char_masks = SegmentationCharMask(
            charsbbs, words=words, use_char_ann=use_char_ann, size=image_shape, char_num_classes=len(self.char_classes)
        )
        target.add_field("char_masks", char_masks)
        return image, target
    
    def get(self):
        images_ls, target_ls = self.generate()
        for img, target in zip(images_ls, target_ls):
            yield img, target

    def form_dataset(self):
        datacomb = dataset.GeneratorDataset(self.get, column_names=["image", "target"])
        return datacomb
    
    def get_size(self):
        return self.size



if __name__ == '__main__':
    import pickle as pkl
    dm = DatasetsManager(config=config)
    with open('datasets/icdar2013/train_images/100.jpg', 'rb') as f:
        img = f.read()
    with open('datasets/icdar2013/train_gts/100.jpg.txt', 'rb') as f:
        gt = f.read()
    transferred = dm.transfer(img, gt)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/img.pkl', 'wb') as f:
        pkl.dump(transferred[0], f)
    # words, boxes, charsbbs, segmentations, labels
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/words.pkl', 'wb') as f:
        pkl.dump(transferred[1], f)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/boxes.pkl', 'wb') as f:
        pkl.dump(transferred[2], f)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/charbboxs.pkl', 'wb') as f:
        pkl.dump(transferred[3], f)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/segmentations.pkl', 'wb') as f:
        pkl.dump(transferred[4], f)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/labels.pkl', 'wb') as f:
        pkl.dump(transferred[5], f)
    _, target = dm.restore_dataset(img, gt)
    with open('/home/tiantian/Documents/MaskTextSpotter-MindSpore/unittest/case/target.pkl', 'wb') as f:
        pkl.dump(target, f)
