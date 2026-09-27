import math

def get_lr(step, config):
    if step < config.warmup_steps:
        return config.lr_max * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.lr_max * config.min_lr_ratio

    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    lr_min = config.lr_max * config.min_lr_ratio
    return lr_min + coeff * (config.lr_max - lr_min)