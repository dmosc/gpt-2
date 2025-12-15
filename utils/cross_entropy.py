import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits -= logits.max(dim=-1, keepdim=True).values
    exp_logits = logits.exp()
    sum_exp = exp_logits.sum(dim=-1)
    target_exp = exp_logits.gather(dim=-1,
                                    index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -(sum_exp / target_exp).log()
    return loss.mean()