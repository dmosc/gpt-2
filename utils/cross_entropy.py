import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    *batch_dims, vocab_size = logits.shape