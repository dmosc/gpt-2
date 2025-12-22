import torch
import math

from collections.abc import Callable, Iterable
from typing import Optional


class DecayedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr < 0, f'{lr=} must be > 0.'
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        for param_group in self.param_groups:
            lr = param_group['lr']
            for param in param_group['params']:
                if param.grad is not None:
                    param_state = self.state[param]
                    curr_step = param_state.get('curr_step', 0)
                    param.data -= lr / \
                        math.sqrt(curr_step + 1) * param.grad.data
                    param_state['curr_step'] = curr_step + 1
        return loss
