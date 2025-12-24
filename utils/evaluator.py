import torch


class Metric:
    def __init__(self, name: str):
        self.name = name
        self.values = []

    def update(self, value: float):
        self.values.append(value)

    def get_latest(self) -> float | None:
        return self.values[-1] if self.values else None


class Evaluator:
    def __init__(self, leading_indicator: str = 'cross_entropy'):
        self.metrics: dict[str, Metric] = {
            'cross_entropy': Metric('cross_entropy'),
            'perplexity': Metric('perplexity')
        }
        self.leading_indicator = leading_indicator
        self.log_every_n_steps = 100

    def evaluate(self, step: int, logits: torch.Tensor, targets: torch.Tensor):
        cross_entropy = self._cross_entropy(logits, targets)
        perplexity = self._perplexity(cross_entropy.mean()).item()
        self.metrics['cross_entropy'].update(cross_entropy)
        self.metrics['perplexity'].update(perplexity)
        if step % self.log_every_n_steps == 0:
            print(f'{step=} {cross_entropy.item()=}, {perplexity=}')
        return self._pick_leading_metric().get_latest()

    def _cross_entropy(self, logits: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        logits_logsumexp = torch.logsumexp(logits, dim=-1)
        target_logits = torch.gather(logits, dim=-1,
                                     index=targets.unsqueeze(-1)).squeeze(-1)
        return (logits_logsumexp - target_logits).mean()

    def _perplexity(self, mean_cross_entropy: torch.Tensor) -> torch.Tensor:
        return mean_cross_entropy.exp()

    def _pick_leading_metric(self) -> Metric:
        return self.metrics[self.leading_indicator]
