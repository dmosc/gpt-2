import torch
import math


class CosAnnealingScheduler:
    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int,
                 max_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int):
        if step < self.warmup_steps:
            return step * self.max_lr / self.warmup_steps
        if self.warmup_steps <= step < self.max_steps:
            return self.min_lr + self._cos_annealing(step)
        return self.min_lr

    def _cos_annealing(self, step):
        steps_diff = self.max_steps - self.warmup_steps
        lr_diff = self.max_lr - self.min_lr
        cos = math.cos((step - self.warmup_steps) / (steps_diff) * math.pi)
        return 0.5 * (1 + cos) * lr_diff
