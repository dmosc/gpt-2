import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = 2 / (in_features + out_features)
        weights = torch.rand(size=(out_features, in_features), dtype=dtype)
        weights = torch.nn.init.trunc_normal_(weights,
                                              mean=0, std=std,
                                              a=-3 * std ** 0.5,
                                              b=3 * std ** 0.5)
        self.weights = torch.nn.Parameter(weights)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x.to(self.device), self.weights.to(self.device),
                      '... d_in, d_out d_in -> ... d_out')
