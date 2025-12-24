import torch

from modules import RMSNorm, CausalSelfAttn, FeedForward


class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int) -> None:
        super().__init__()
        self.causal_self_attn_prenorm = RMSNorm(d_model)
        self.causal_self_attn = CausalSelfAttn(d_model, num_heads, max_seq_len)
        self.feed_forward_prenorm = RMSNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.causal_self_attn(self.causal_self_attn_prenorm(x))
        x = x + self.feed_forward(self.feed_forward_prenorm(x))
        return x
