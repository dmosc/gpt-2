import torch

from modules import Embedding, Transformer, RMSNorm, Linear
from utils import softmax


class LanguageModel(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int,
                 max_seq_len: int, num_layers: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.layers = self._init_layers(d_model, num_heads, d_ff, max_seq_len,
                                        num_layers, vocab_size)

    def generate_next_token(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_logits(tokens)[..., -1, :].argmax(dim=-1)

    def _init_layers(self, d_model: int, num_heads: int, d_ff: int,
                     max_seq_len: int, num_layers: int,
                     vocab_size: int) -> torch.nn.Sequential:
        layers = torch.nn.Sequential(Embedding(vocab_size, d_model))
        for _ in range(num_layers):
            layers.append(Transformer(d_model, num_heads, d_ff, max_seq_len))
        layers.append(RMSNorm(d_model))
        layers.append(Linear(d_model, vocab_size))
        return layers

    def _compute_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        *batch_size, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, f'{seq_len=} exceeds {self.max_seq_len=}.'
        logits = softmax(self.layers(tokens), -1)
        return logits
