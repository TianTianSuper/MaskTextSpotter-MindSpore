from bisect import bisect_right
import math

def warmup_lr(config, rank_size=1, start_steps=0):
    base_lr = config.base_lr
    gamma = config.lr_gamma
    milestone = config.lr_steps
    base_step = (config.base_step // rank_size) + rank_size
    total_steps = int(base_step * config.total_epoch)
    warmup_steps = int(config.warmup_step)
    factor = config.warmup_factor
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            if config.warmup_method == "constant":
                warmup_factor = config.warmup_factor
            elif config.warmup_method == "linear":
                alpha = i / warmup_steps
                warmup_factor = factor * (1 - alpha) + alpha
            else:
                raise ValueError("Warmup method should be 'constant' or 'linear', got {}".format(warmup_factor))

        lr.append(base_lr * warmup_factor * gamma ** bisect_right(milestone, i))
    learning_rate = lr[start_steps:]
    return learning_rate