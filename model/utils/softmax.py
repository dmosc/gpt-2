import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not x.is_floating_point():
        x = x.float()
    max_val = x.max(dim=dim, keepdim=True).values
    shifted_x = x - max_val
    shifted_x_exp = shifted_x.exp()
    return shifted_x_exp / shifted_x_exp.sum(dim=dim, keepdim=True)
