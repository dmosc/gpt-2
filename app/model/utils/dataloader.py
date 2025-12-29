import os
import torch
import random
import itertools

from collections import deque

from pathlib import Path
from typing import Iterator

from ..config import Config
from ..utils.text_tokenizer import Tokenizer


class DataLoader:
    def __init__(self, config: Config, tokenizer: Tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.data = self._load_data(self.config.train_data_path)

    def get_next_batch(self,
                       randomize: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        max_tokens = self.config.batch_size * (self.config.seq_len + 1)
        if randomize:
            skip_num_examples = random.randint(
                0, self.config.skip_up_to_n_batches)
            # Consume the next skip_num_examples items from the data iterator.
            deque(itertools.islice(self.data, skip_num_examples), maxlen=0)
        data = next(iter(self.data), None)
        if data is None:
            # Reinitialize the pointer to iterate over all the data again.
            self.data = self._load_data(self.config.train_data_path)
            print('Finished processing all data; resetting pointer to start.')
            return None
        tokens = torch.tensor(self.tokenizer.encode(data)[:max_tokens])
        batch = torch.stack([tokens[i: i + self.config.seq_len]
                            for i in range(self.config.batch_size)])
        targets = torch.stack([tokens[i + 1: i + 1 + self.config.seq_len]
                              for i in range(self.config.batch_size)])
        return batch, targets

    def _load_data(self, path: Path) -> Iterator[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path=} doesn\'t exist.')
        with open(path, 'rb') as file:
            # 4-bytes is the max size for a single UTF-8 character which is how
            # we're interpreting the file contents.
            chunk_bytes = self.config.batch_size * self.config.seq_len * 4
            while chunk := file.read(chunk_bytes):
                yield chunk.decode('utf-8', errors='ignore')
