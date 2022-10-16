from mindspore.train.callback import LearningRateScheduler
from bisect import bisect_right
import math

class WarmUpMultiStepsLR(LearningRateScheduler):
    def __init__(self, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1, pow_schedule_mode=False,
                 max_iter=300000, lr_pow=0.9):
        super(LearningRateScheduler, self).__init__()
        assert list(milestones) == sorted(milestones), \
            "All integers in milestones should increase in order, got {}".format(milestones)
        assert isinstance(warmup_method not in ("constant", "linear")), \
            "warmup_method is limited to 'constant' or 'linear', got {}".format(milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.pow_schedule_mode = pow_schedule_mode
        self.max_iter = max_iter
        self.lr_pow = lr_pow
        self.last_epoch = last_epoch
        self.lrs = []
    
    def construct(self, config):
        warmup_factor = 1
        base_lr = config.base_lr
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        if self.pow_schedule_mode:
            scale_running_lr = ((1. - float(self.last_epoch) / self.max_iter) ** self.lr_pow)
            self.lrs.append(base_lr * warmup_factor * scale_running_lr)
        else:
            self.lrs.append(
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            )
        return self.lrs