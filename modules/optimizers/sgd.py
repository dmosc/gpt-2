import torch
import math

from collections.abc import Callable, Iterable
from typing import Optional


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr > 0, f'{lr=} should be > 0'
        defaults = {'lr': lr}
        super().__init__(params, defaults)
        self.step_i = 0

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        self.step_i += 1
        for param_group in self.param_groups:
            lr = param_group['lr']
            alpha = -lr / math.sqrt(self.step_i)
            for param in param_group['params']:
                if param.grad is not None:
                    param.data.add_(param.grad, alpha=alpha)
        return loss
