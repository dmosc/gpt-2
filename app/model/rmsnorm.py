import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None,
                 dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.gain = torch.nn.Parameter(torch.ones((d_model,), dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self._rms(x)) * self.gain

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
