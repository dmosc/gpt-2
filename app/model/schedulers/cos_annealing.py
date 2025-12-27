import math
from ..config import Config


class CosAnnealingScheduler:
    def __init__(self, config: Config) -> None:
        self.config = config

    def get_lr(self, step: int):
        if step < self.config.warmup_steps:
            return step * self.config.max_lr / self.config.warmup_steps
        if self.config.warmup_steps <= step < self.config.max_steps:
            return self.config.min_lr + self._cos_annealing(step)
        return self.config.min_lr

    def _cos_annealing(self, step):
        steps_diff = self.config.max_steps - self.config.warmup_steps
        lr_diff = self.config.max_lr - self.config.min_lr
        cos = math.cos((step - self.config.warmup_steps) /
                       (steps_diff) * math.pi)
        return 0.5 * (1 + cos) * lr_diff
