import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets)


def perplexity(mean_cross_entropy: torch.Tensor) -> torch.Tensor:
    return mean_cross_entropy.exp()