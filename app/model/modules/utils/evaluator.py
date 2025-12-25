import torch


class Metric:
    def __init__(self, name: str):
        self.name = name
        self.values: list[torch.Tensor] = []

    def update(self, value: torch.Tensor):
        self.values.append(value)

    def get_latest(self) -> torch.Tensor | None:
        return self.values[-1] if self.values else None


class Evaluator:
    def __init__(self, leading_indicator: str = 'cross_entropy',
                 log_every_n_steps: int = 100):
        self.metrics: dict[str, Metric] = {
            'cross_entropy': Metric('cross_entropy'),
            'perplexity': Metric('perplexity')
        }
        self.leading_indicator = leading_indicator
        self.log_every_n_steps = log_every_n_steps

    def evaluate(self, step: int, logits: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = self._cross_entropy(logits, targets)
        perplexity = self._perplexity(cross_entropy.mean())
        self.metrics['cross_entropy'].update(cross_entropy)
        self.metrics['perplexity'].update(perplexity)
        if step % self.log_every_n_steps == 0:
            print(f'{step=} {cross_entropy.item()=}, {perplexity.item()=}')
        if latest_measurement := self._pick_leading_metric().get_latest():
            return latest_measurement
        raise ValueError('No leading metric measurement available.')

    @staticmethod
    def load_state_dict(state_dict: dict) -> 'Evaluator':
        evaluator = Evaluator(leading_indicator=state_dict['leading_indicator'])
        evaluator.log_every_n_steps = state_dict['log_every_n_steps']
        for name, values in state_dict.get('metrics', {}).items():
            evaluator.metrics[name].values = [torch.tensor(v) for v in values]
        return evaluator

    def state_dict(self) -> dict:
        return {
            'leading_indicator': self.leading_indicator,
            'log_every_n_steps': self.log_every_n_steps,
            'metrics': {
                name: [v.item() for v in metric.values]
                for name, metric in self.metrics.items()
            }
        }

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
