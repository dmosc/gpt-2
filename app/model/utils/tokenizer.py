import os

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from .perf_utils import time_func


class Tokenizer(ABC):
    default_chunk_size_bytes = 100_000  # 100 KB
    default_parallel_processes = os.cpu_count()
    pretokens_path_prefix = Path('/tmp/gpt-tmp')
    vocab_file_path = pretokens_path_prefix / 'vocab.pkl'
    vocab = None

    @property
    def reverse_vocab(self):
        """
        Returns self.vocab with keys and values reversed.
        """
        if hasattr(self, '_reverse_vocab'):
            return self._reverse_vocab
        if self.vocab:
            self._reverse_vocab = {v: k for v, k in enumerate(self.vocab)}
            return self._reverse_vocab
        else:
            raise ValueError(
                'Tokenizer hasn\'t been initialized. Run .train().')

    @time_func
    @abstractmethod
    def load(self, vocab_file_path: Optional[Path] = None) -> None:
        """
        Loads the vocabulary from disk.

        Args:
            vocab_file_path (Path): Path to the vocabulary file.
        """
        pass

    @time_func
    @abstractmethod
    def encode(self, sequence: str) -> list[int]:
        """
        Encodes the input sequence into a list of token IDs using the trained vocabulary.

        Args:
            sequence (str): The input sequence to encode.
        Returns:
            list[int]: List of token IDs.
        """
        pass

    @time_func
    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a sequence of bytes into a UTF-8 Unicode string by joining
        tokens from the learned vocabulary.

        Args:
            tokens (list[int]): The list of tokens to decode.

        Returns:
            str: String decoded in UTF-8 Unicode format.
        """
        pass

    @time_func
    @abstractmethod
    def train(self, dataset_path: Path, max_vocab_size: int,
              special_tokens: list[bytes]) -> None:
        """
        Trains the tokenizer on the given dataset to build a vocabulary.

        Args:
            dataset_path (Path): Path to the training dataset file.
            max_vocab_size (int): Maximum size of the vocabulary.
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        pass
