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
    document_split_token = None

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

    def _get_dataset_chunk_offsets(self, dataset_path: Path) -> list[int]:
        """
        Find indices to appropriately split a provided dataset file using the
        document_split_token value to semantically separate documents being
        ingested from the same dataset_path source.

        Args:
            dataset_path (Path): Path to the training dataset file.
        """
        with open(dataset_path, 'rb') as file:
            dataset_size_bytes = os.path.getsize(dataset_path)
            chunk_size_bytes = min(
                self.default_chunk_size_bytes, dataset_size_bytes)
            chunk_count = dataset_size_bytes // chunk_size_bytes
            # Initial, evenly spaced chunk offsets
            chunk_offsets: list[int] = []
            for i in range(chunk_count):
                chunk_offsets.append(i * chunk_size_bytes)
            chunk_offsets.append(dataset_size_bytes)
            # Potentially cut short or extend chunks to nearest
            # document_split_token to align chunks with text boundaries
            #
            # Look ahead 4 KiB at a time to find the nearest split token
            look_ahead_size = 4 * 1024
            with open(dataset_path, 'rb') as file:
                for i in range(1, len(chunk_offsets) - 1):
                    curr_offset = chunk_offsets[i]
                    file.seek(curr_offset)
                    while True:
                        chunk = file.read(look_ahead_size)
                        if not chunk:
                            break
                        split_token_idx = chunk.find(
                            self.document_split_token) if self.document_split_token else -1
                        if split_token_idx == -1:
                            curr_offset += look_ahead_size
                        else:
                            chunk_offsets[i] += split_token_idx
                            break
            return sorted(set(chunk_offsets))
