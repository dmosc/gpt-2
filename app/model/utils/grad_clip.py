import torch

from torch.nn.parameter import Parameter
from typing import Iterator


def grad_clip(params: Iterator[Parameter], max_norm: float,
              eps: float = 1e-6) -> torch.Tensor:
    l2_norm = 0.
    for param in params:
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()
            l2_norm += param_norm ** 2
    l2_norm = l2_norm ** 0.5
    if l2_norm > max_norm:
        clip_coeff = max_norm / (l2_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(clip_coeff)
    return l2_norm
