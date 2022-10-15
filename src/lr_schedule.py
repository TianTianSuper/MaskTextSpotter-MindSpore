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
    
    def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
        lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
        learning_rate = float(init_lr) + lr_inc * current_step
        return learning_rate

    def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
        base = float(current_step - warmup_steps) / float(decay_steps)
        learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
        return learning_rate
    
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
            return [
                base_lr * warmup_factor * scale_running_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]