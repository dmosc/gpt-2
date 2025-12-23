import torch


class Evaluator:
    def cross_entropy(self, logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        logits_logsumexp = torch.logsumexp(logits, dim=-1)
        target_logits = torch.gather(logits, dim=-1,
                                     index=targets.unsqueeze(-1)).squeeze(-1)
        return (logits_logsumexp - target_logits).mean()

    def perplexity(self, mean_cross_entropy: torch.Tensor) -> torch.Tensor:
        return mean_cross_entropy.exp()
