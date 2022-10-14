from bisect import bisect_right

def pre_computed_lr(config, rank_size=1, start_steps=0):
    base_lr = config.base_lr
    static_warmup_factor = config.warmup_factor
    static_warmup_iters = config.warmup_iters
    static_warmup_method = config.warmup_method
    static_lr_gamma = config.lr_gamma
    static_milestones = config.lr_milestones

    assert list(static_milestones) == sorted(static_milestones), \
        "All integers in milestones should increase in order, got {}".format(static_milestones)
    assert isinstance(static_warmup_method not in ("constant", "linear")), \
        "warmup_method is limited to 'constant' or 'linear', got {}".format(static_warmup_method)

    base_step = (config.base_step // rank_size) + rank_size
    total_steps = int(base_step * config.total_epoch)
    warmup_steps = int(config.warmup_step)
    lr = []

    for i in range(total_steps):
        if i < warmup_steps:
            if static_warmup_method == 'constant':
                warmup_factor = static_warmup_factor
            elif static_warmup_factor == 'linear':
                ratio = i / static_warmup_iters
                warmup_factor = static_warmup_factor * (1 - ratio) + ratio
        lr.append(base_lr * warmup_factor * static_lr_gamma ** bisect_right(static_milestones, i))
    return lr[start_steps:]