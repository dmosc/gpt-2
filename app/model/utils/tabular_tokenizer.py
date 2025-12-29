import pickle

from pathlib import Path
from typing import Optional

from .tokenizer import Tokenizer


class TabularTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def load(self, vocab_file_path: Optional[Path] = None) -> None:
        """
        Loads the vocabulary from disk.

        Args:
            vocab_file_path (Path): Path to the vocabulary file.
        """
        if vocab_file_path is None:
            vocab_file_path = self.vocab_file_path

        with open(vocab_file_path, 'rb') as file:
            self.vocab = pickle.load(file)

    def train(self, dataset_path: Path, max_vocab_size: int,
              special_tokens: list[bytes]):
        chunk_offsets = self._get_dataset_chunk_offsets(dataset_path)