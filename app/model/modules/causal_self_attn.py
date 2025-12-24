import torch

from .utils import softmax
from .linear import Linear
from .rope import RoPE


class CausalSelfAttn(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int) -> None:
        assert d_model % num_heads == 0, f'{d_model=} should be divisible by {num_heads=}'
        super().__init__()
        self.num_heads = num_heads
        self.d_qk = d_model // num_heads
        self.d_v = self.d_qk
        self.w_q = Linear(d_model, self.d_qk * num_heads)
        self.w_k = Linear(d_model, self.d_qk * num_heads)
        self.w_v = Linear(d_model, self.d_v * num_heads)
        self.w_o = Linear(self.d_v * num_heads, d_model)
        self.rope = RoPE(10_000, self.d_qk, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, seq_len, d_model = x.shape
        assert d_model == self.d_qk * self.num_heads, \
            f'{d_model=} must equal {self.d_qk * self.num_heads}'
        positions = torch.arange(seq_len,
                                 device=x.device).reshape(1, 1, seq_len)
        q = self.w_q(x).reshape(
            *batch_dims, self.num_heads, seq_len, self.d_qk)
        rotated_q = self.rope(q, positions)
        k = self.w_k(x).reshape(
            *batch_dims, self.num_heads, seq_len, self.d_qk)
        rotated_k = self.rope(k, positions)
        v = self.w_v(x).reshape(*batch_dims, self.num_heads, seq_len, self.d_v)
        mask = self._causal_mask(seq_len=seq_len)
        scaled_attn = self._scaled_dot_product_attn(rotated_q, rotated_k, v,
                                                    mask)
        concat_attn = scaled_attn.reshape(*batch_dims, seq_len,
                                          self.d_v * self.num_heads)
        return self.w_o(concat_attn)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        # Creates a lower triangular mask where token at index i pays attention
        # to token at index j if the value is True. The lower triangular shape
        # makes it so that i can only attend to positions j <= i.
        return torch.tril(torch.ones(seq_len, seq_len)).bool()

    def _scaled_dot_product_attn(self, q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_qk ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)
