import torch

from utils import softmax


class CausalSelfAttn(torch.nn.Module):
    def __init__(self, d_qk: int, d_v: int) -> None:
        super().__init__()
        self.d_qk = d_qk
        self.d_v = d_v

    def _scaled_dot_product_attn(self, q: torch.Tensor, k: torch.Tensor,
                                 v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_qk ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == False, float('-inf'))
        return torch.matmul(torch.softmax(attn_scores, dim=-1), V)
