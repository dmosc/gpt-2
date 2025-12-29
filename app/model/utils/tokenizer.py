import os
import pickle

from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any
from abc import ABC, abstractmethod

from .perf_utils import time_func


class Tokenizer(ABC):
    default_chunk_size_bytes = 100_000  # 100 KB
    default_parallel_processes = os.cpu_count()
    tmp_dir_path = Path('/tmp/gpt-tmp')
    vocab_file_path = tmp_dir_path / 'vocab.pkl'
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
        Loads the vocabulary from disk. Use this method to initialize
        self.vocab.

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
    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a sequence of bytes into a UTF-8 Unicode string by joining
        tokens from the learned vocabulary.

        Args:
            tokens (list[int]): The list of tokens to decode.

        Returns:
            str: String decoded in UTF-8 Unicode format.
        """
        sequence_bytes = b''
        for token in tokens:
            sequence_bytes += self.reverse_vocab[token]
        return sequence_bytes.decode('utf-8', errors='replace')

    @time_func
    def train(self, dataset_path: Path, max_vocab_size: int,
              special_tokens: list[bytes]) -> None:
        """
        Trains the tokenizer on the given dataset to build a vocabulary.

        Args:
            dataset_path (Path): Path to the training dataset file.
            max_vocab_size (int): Maximum size of the vocabulary.
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        self._process_chunks(dataset_path)
        token_freqs = self._get_token_freqs(special_tokens)
        vocab_payload = self._build_vocab(max_vocab_size,
                                          special_tokens, token_freqs)
        self._save_vocab(vocab_payload)
        self._flush_pretokens()

    def _process_chunks(self, dataset_path: Path):
        """
        Applies _process_chunk(...) to multiple dataset chunks in parallel.

        Args:
            dataset_path (Path): Path to the training dataset file.
        """
        chunk_offsets = self._get_dataset_chunk_offsets(dataset_path)
        iterable_args = zip([dataset_path] * (len(chunk_offsets) - 1),
                            chunk_offsets[:-1], chunk_offsets[1:])
        print(f'Tokenizer: Tokenizing {len(chunk_offsets) - 1} chunks '
              f'using {self.default_parallel_processes} parallel processes...')
        with Pool(processes=self.default_parallel_processes) as pool:
            pool.starmap(self._process_chunk, iterable_args)

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

    @abstractmethod
    def _process_chunk(self, dataset_path: Path, start_offset: int,
                       end_offset: int):
        """
        Pretokenizes a chunk region from dataset_path delimited by start_offset
        and end_offset pointers.

        Args:
            dataset_path (Path): Path to the training dataset file.
            start_offset (int): Starting index to initialize the file reading.
            end_offset (int): Final index to finish the file reading.
        """
        raise NotImplementedError(
            'You must implement _process_chunk to process the dataset.')

    def _get_token_freqs(self,
                         special_tokens: list[bytes]) -> dict[tuple[bytes], int]:
        """
        Load pretoken and compute frequencies across all files.

        Args:
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        token_files = list(self.tmp_dir_path.glob('pretokens_*.pkl'))
        token_freqs: dict[tuple[bytes], int] = defaultdict(int)
        for tokens_file in token_files:
            with open(tokens_file, 'rb') as file:
                for token in pickle.load(file):
                    if token in special_tokens:
                        continue
                    token_freqs[tuple(token)] += 1
        return token_freqs

    @abstractmethod
    def _build_vocab(self, max_vocab_size: int,
                     special_tokens: list[bytes],
                     token_freqs: dict[tuple[bytes], int]) -> tuple[Any, ...]:
        """
        Builds the vocabulary.

        Args:
            max_vocab_size (int): Maximum size of the vocabulary.
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        raise NotImplementedError(
            'You must implement _build_vocab to process input sequences.')

    def _save_vocab(self, vocab_payload: tuple[Any, ...]):
        """
        Saves the vocabulary to disk.

        Args:
            vocab_payload: An arbitrary-sized tuple of objects to save.
        """
        print('Tokenizer: Saving final vocabulary...')
        with open(self.vocab_file_path, 'wb') as file:
            pickle.dump(vocab_payload, file)

    def _flush_pretokens(self):
        """
        Deletes temporary pretoken files to free up space.
        """
        print('Tokenizer: Cleaning up temporary pretoken files...')
        pretoken_files = list(self.tmp_dir_path.glob('pretokens_*.pkl'))
        for pretokens_file in pretoken_files:
            os.remove(pretokens_file)
