import torch

from .embedding import Embedding
from .transformer import Transformer
from .rmsnorm import RMSNorm
from .linear import Linear
from .config import Config


class LanguageModel(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.layers = self._init_layers(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f'{seq_len=} exceeds {self.max_seq_len=}.'
        logits = self.layers(x)
        return logits

    def generate_next_token(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        # Get last token's logits and find argmax
        return logits[..., -1, :].argmax(dim=-1)

    def _init_layers(self, config: Config) -> torch.nn.Sequential:
        layers = torch.nn.Sequential(Embedding(config.max_vocab_size,
                                               config.d_model))
        for _ in range(config.num_layers):
            layers.append(Transformer(config.d_model,
                                      config.num_heads,
                                      config.d_ff,
                                      config.max_seq_len))
        layers.append(RMSNorm(config.d_model))
        layers.append(Linear(config.d_model, config.max_vocab_size))
        return layers
