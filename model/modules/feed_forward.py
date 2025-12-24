import torch
from modules import linear

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.linear1 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = linear.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear3 = linear.Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(x)
    
    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self._silu(self.linear1(x)) * self.linear2(x))

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))
