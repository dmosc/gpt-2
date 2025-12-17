import torch

from collections.abc import Callable, Iterable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        assert lr > 0, f'{lr=} should be > 0.'
        assert 0 <= betas[0] < 1, f'{betas[0]=} should be between 0 and 1.'
        assert 0 <= betas[1] < 1, f'{betas[1]=} should be between 0 and 1.'
        assert eps > 0, f'{eps=} should be > 0.'
        assert weight_decay >= 0, f'{weight_decay=} should be >= 0.'
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        for param_group in self.param_groups:
            lr = param_group['lr']
            beta1, beta2 = param_group['betas']
            eps = param_group['eps']
            weight_decay = param_group['weight_decay']
            for param in param_group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    if len(state) == 0:
                        state['t'] = 0
                        state['m'] = torch.zeros_like(param.data)
                        state['v'] = torch.zeros_like(param.data)
                    state['t'] += 1
                    state['m'] = beta1 * state['m'] + (1 - beta1) * param.grad
                    state['v'] = beta2 * state['v'] + \
                        (1 - beta2) * (param.grad ** 2)
                    # Apply weight decay first
                    if weight_decay != 0:
                        param.data.mul_(1 - lr * weight_decay)
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['t']
                    bias_correction2 = 1 - beta2 ** state['t']
                    # Compute step size
                    step_size = lr / bias_correction1
                    bias_corrected_v = state['v'] / bias_correction2
                    # Update parameters
                    param.data.add_(state['m'] / (bias_corrected_v.sqrt() + eps),
                                    alpha=-step_size)
