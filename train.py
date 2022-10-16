import os

from src.model_utils.config import config
from src.general import MaskTextSpotter3, GeneralLoss
from src.network_define import LossCallBack
from src.dataset.generator import DatasetsManager
from src.lr_schedule import WarmUpMultiStepsLR

import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum, TrainOneStepCell, WithLossCell
from mindspore.common import set_seed

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def train_masktextspotter():
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=get_device_id())

    print('\ntrain.py config:\n', config)
    print("Start train for maskrcnn!")

    dataset_sink_mode_flag = True
    if not config.do_eval and config.run_distribute:
        init()
        rank = get_rank()
        dataset_sink_mode_flag = device_target == 'Ascend'
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    prefix = "MaskTextSpotter.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        dm = DatasetsManager(config)
        dataset = dm.init_dataset()
    dataset_size = dataset.get_dataset_size()
    
    net = MaskTextSpotter3(config)
    net.set_train(True)

    loss = GeneralLoss()
    lr = WarmUpMultiStepsLR(config)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    net_with_loss = WithLossCell(net, loss)
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    # callbacks
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cbs = [time_cb, loss_cb]    
    if config.save_ckpt:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                        keep_checkpoint_max=config.keep_checkpoint_max)
        if not os.path.exists(config.save_checkpoint_path):
            os.mkdir(config.save_checkpoint_path)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cbs += [ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=dataset_sink_mode_flag)

if __name__ == '__main__':
    train_masktextspotter()
