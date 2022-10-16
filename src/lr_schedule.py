from mindspore.train.callback import LearningRateScheduler
from bisect import bisect_right
import math

class WarmUpMultiStepsLR(LearningRateScheduler):
    def __init__(self, config, gamma=0.1, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1, pow_schedule_mode=False,
                 lr_pow=0.9):

        self.gamma = gamma
        self.milestone = config.lr_steps
        self.warmup_factor = config.warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.pow_schedule_mode = pow_schedule_mode
        self.max_iter = config.max_iter
        self.lr_pow = lr_pow
        self.last_epoch = last_epoch
        self.base_lr = config.base_lr
        super(LearningRateScheduler, self).__init__(self.update)
    
    def update(self):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Warmup method should be 'constant' or 'linear', got {}".format(self.warmup_factor))
        if self.pow_schedule_mode:
            scale_running_lr = ((1. - float(self.last_epoch) / self.max_iter) ** self.lr_pow)
            return self.base_lr * warmup_factor * scale_running_lr
        else:
            return self.base_lr * warmup_factor * self.gamma ** bisect_right(self.milestone, self.last_epoch)
