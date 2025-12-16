import os
import torch

from pathlib import Path
from typing import Iterator

from .tokenizer import Tokenizer


class DataLoader:
    def __init__(self, tokenizer: Tokenizer, path: Path, batch_size: int,
                 seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.data = self._load_data(path)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_tokens = self.batch_size * self.seq_len + self.batch_size
        tokens = torch.tensor(
            self.tokenizer.encode(next(self.data))[:max_tokens])
        batch = torch.stack([tokens[i: i + self.seq_len]
                            for i in range(self.batch_size)])
        targets = tokens[self.seq_len: self.seq_len + self.batch_size]
        return batch, targets

    def _load_data(self, path: Path) -> Iterator[bytes]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path=} doesn\'t exist.')
        with open(path, 'rb') as file:
            # 4-bytes is the max size for a single UTF-8 character which is how
            # we're interpreting the file contents.
            chunk_bytes = self.batch_size * self.seq_len * 4
            while chunk := file.read(chunk_bytes):
                yield chunk.decode('utf-8', errors='ignore')
