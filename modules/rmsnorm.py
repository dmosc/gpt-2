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
        # Upcast tensor to prevent overflow when squaring the input.
        original_dtype = x.dtype
        x.to(torch.float32)
        x.to(self.device)
        # Return back to original dtype.
        return (x / self._rms(x) * self.gain).to(original_dtype)
    
    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((torch.sum(x ** 2) + self.eps) / self.d_model)
