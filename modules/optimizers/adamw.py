import torch
import math

from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        assert lr > 0, f'{lr=} must be > 0.'
        assert 0 <= betas[0] < 1, f'{betas[0]=} must be between 0 and 1.'
        assert 0 <= betas[1] < 1, f'{betas[1]=} must be between 0 and 1.'
        assert eps > 0, f'{eps=} must be > 0.'
        assert weight_decay >= 0, f'{weight_decay=} must be >= 0.'
        defaults = {'lr': lr, 'betas': betas, 'eps': eps,
                    'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        for param_group in self.param_groups:
            lr = param_group['lr']
            betas = param_group['betas']
            eps = param_group['eps']
            weight_decay = param_group['weight_decay']
            for param in param_group['params']:
                if param.grad is not None:
                    param_state = self.state[param]
                    if len(param_state) == 0:
                        param_state['curr_step'] = 0
                        param_state['moment1'] = torch.zeros_like(param.data)
                        param_state['moment2'] = torch.zeros_like(param.data)
                    # Update state params.
                    param_state['curr_step'] += 1
                    param_state['moment1'] = betas[0] * \
                        param_state['moment1'] + (1 - betas[0]) * param.grad
                    param_state['moment2'] = betas[1] * \
                        param_state['moment2'] + \
                        (1 - betas[1]) * (param.grad ** 2)
                    # Compute adjusted learning rate based on moment vectors.
                    adjusted_lr = lr * (math.sqrt(1 - betas[1] ** param_state['curr_step'])) / (
                        1 - betas[0] ** param_state['curr_step'])
                    # Update params.
                    param.data -= adjusted_lr * \
                        param_state['moment1'] / \
                        (param_state['moment2'] ** 0.5 + eps)
                    # Apply weight decay.
                    param.data -= lr * weight_decay * param.data
