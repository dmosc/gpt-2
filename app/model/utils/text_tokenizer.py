import os
import re
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Optional

from .perf_utils import time_func


class Tokenizer:
    default_chunk_size_bytes = 100_000  # 100 KB
    default_parallel_processes = os.cpu_count()
    # OpenAI's Tiktoken pre-tokenizer regex pattern:
    # https://github.com/openai/tiktoken/pull/234/files
    pretokenizer_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\s\\p{L}\\p{N}]+|\s+(?!\S)|\s+"""
    pretokens_path_prefix = Path('/tmp/gpt-2')
    vocab_file_path = pretokens_path_prefix / 'vocab.pkl'
    end_of_chunk_split_token = b'<|endoftext|>'

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
    def load(self, vocab_file_path: Optional[Path] = None) -> None:
        """
        Loads the vocabulary from disk.

        Args:
            vocab_file_path (Path): Path to the vocabulary file.
        """
        if vocab_file_path is None:
            vocab_file_path = self.vocab_file_path

        with open(vocab_file_path, 'rb') as file:
            self.vocab, self.merges = pickle.load(file)

    @time_func
    def encode(self, sequence: str) -> list[int]:
        """
        Encodes the input sequence into a list of token IDs using the trained vocabulary.

        Args:
            sequence (str): The input sequence to encode.
        Returns:
            list[int]: List of token IDs.
        """
        if not hasattr(self, 'vocab') or not hasattr(self, 'merges'):
            raise ValueError(
                'Tokenizer vocabulary not loaded. Call load() first.')

        pretokens = self._pretokenize_sequence(sequence)
        all_tokens: list[int] = []
        for pretoken in pretokens:
            tokens = [self.vocab[bytes([t])] for t in pretoken]
            for merge in self.merges:
                idx = 0
                existing_merge = merge[0] + merge[1]
                while idx < len(tokens) - 1:
                    potential_merge = (
                        self.reverse_vocab[tokens[idx]] +
                        self.reverse_vocab[tokens[idx + 1]]
                    )
                    if potential_merge == existing_merge:
                        tokens[idx] = self.vocab[potential_merge]
                        del tokens[idx + 1]
                        idx += 2
                    else:
                        idx += 1
            all_tokens.extend(tokens)
        return all_tokens

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
        # Pre-tokenize dataset in parallel chunks and save to temporary files.
        chunk_offsets = self._get_dataset_chunk_offsets(dataset_path)
        iterable_args = zip([dataset_path] * (len(chunk_offsets) - 1),
                            chunk_offsets[:-1], chunk_offsets[1:])
        print(f'Tokenizer: Pre-tokenizing {len(chunk_offsets) - 1} chunks '
              f'using {self.default_parallel_processes} parallel processes...')
        with Pool(processes=self.default_parallel_processes) as pool:
            pool.starmap(self._pretokenize_file_chunk, iterable_args)

        # Build vocabulary from pretokenized files
        self.vocab, self.merges = self._build_vocab(max_vocab_size,
                                                    special_tokens)
        self._save_vocab(self.vocab, self.merges)

        # Clean up temporary pretoken files
        self._flush_pretokens()

    def _get_dataset_chunk_offsets(self, dataset_path: Path) -> list[int]:
        """
        Find indices to appropriately split a provided dataset file using the
        end_of_chunk_split_token value to semantically separate documents being
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
            # end_of_chunk_split_token to align chunks with text boundaries
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
                            self.end_of_chunk_split_token)
                        if split_token_idx == -1:
                            curr_offset += look_ahead_size
                        else:
                            chunk_offsets[i] += split_token_idx
                            break
            return sorted(set(chunk_offsets))

    def _pretokenize_file_chunk(self, dataset_path: Path, start_offset: int,
                                end_offset: int) -> None:
        """
        Pretokenizes a chunk region from dataset_path delimited by start_offset
        and end_offset pointers.

        Args:
            dataset_path (Path): Path to the training dataset file.
            start_offset (int): Starting index to initialize the file reading.
            end_offset (int): Final index to finish the file reading.
        """
        # Read chunk from dataset and pretokenize.
        pretokens: list[bytes] = []
        with open(dataset_path, 'rb') as file:
            file.seek(start_offset)
            chunk = file.read(end_offset - start_offset)
            pretokens = self._pretokenize_sequence(
                chunk.decode('utf-8', errors='ignore'))

        # Save pretokens to temporary file.
        pretokens_file_path = self.pretokens_path_prefix / \
            f'pretokens_{start_offset}_{end_offset}.pkl'
        pretokens_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pretokens_file_path, 'wb') as file:
            pickle.dump(pretokens, file)

    def _pretokenize_sequence(self, sequence: str) -> list[bytes]:
        """
        Pretokenizes a single byte sequence using the pretokenizer regex pattern.

        Args:
            sequence (str): The string sequence to pretokenize.

        Returns:
            list[bytes]: List of pretokens.
        """
        pattern = re.compile(self.pretokenizer_pattern)
        matches = pattern.finditer(sequence)
        return [match.group().encode('utf-8') for match in matches]

    def _build_vocab(self, max_vocab_size: int, special_tokens: list[bytes]) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
        """
        Builds the vocabulary from pretokenized files and performs BPE merging.

        Args:
            max_vocab_size (int): Maximum size of the vocabulary.
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        # Load pretoken and compute frequencies across all files.
        pretoken_files = list(
            self.pretokens_path_prefix.glob('pretokens_*.pkl'))
        pretoken_freqs: dict[tuple[bytes], int] = defaultdict(int)
        for pretokens_file in pretoken_files:
            with open(pretokens_file, 'rb') as file:
                for pretoken in pickle.load(file):
                    if pretoken in special_tokens:
                        continue
                    pretoken_freqs[tuple(pretoken)] += 1

        # Initialize vocabulary and reverse vocabulary.
        vocab, reverse_vocab = self._init_vocab(special_tokens)
        # Perform BPE merging to build vocabulary.
        vocab, reverse_vocab, merges = self._bpe_merge(vocab, reverse_vocab,
                                                       pretoken_freqs, max_vocab_size)
        return vocab, merges

    def _init_vocab(self, special_tokens: list[bytes]) -> tuple[dict[bytes, int], dict[int, bytes]]:
        """
        Initializes the vocabulary with single-byte tokens and special tokens.

        Args:
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.

        Returns:
            tuple[dict[bytes, int], dict[int, bytes]]: The initialized vocabulary and reverse vocabulary.
        """
        vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        vocab.update(
            {token: 256 + i for i, token in enumerate(special_tokens)})
        reverse_vocab: dict[int, bytes] = {
            v: k for k, v in vocab.items()}
        return vocab, reverse_vocab

    def _bpe_merge(self, vocab: dict[bytes, int],
                   reverse_vocab: dict[int, bytes],
                   pretoken_freqs: dict[tuple[bytes], int], max_vocab_size: int) -> tuple[dict[bytes, int], dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Performs BPE merging iteratively until max_vocab_size is reached.
        Optimizes frequency calculation by only updating where merges occur.

        Args:
            vocab (dict[bytes, int]): Current vocabulary mapping tokens to IDs.
            reverse_vocab (dict[int, bytes]): Reverse vocabulary mapping IDs to tokens.
            pretoken_freqs (dict[tuple[bytes], int]): Frequencies of pretokens.
            max_vocab_size (int): Maximum size of the vocabulary.

        Returns:
            tuple[dict[bytes, int], dict[int, bytes], list[tuple[bytes, bytes]]]: The final vocabulary, reverse vocabulary, and list of merges.
        """
        merges: list[tuple[bytes, bytes]] = []
        while len(vocab) < max_vocab_size:
            if len(vocab) % 100 == 0:
                print(f'Tokenizer: BPE merging... Current vocab size: '
                      f'{len(vocab)}')
            # Compute byte pair frequencies.
            byte_pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
            for pretoken, freq in pretoken_freqs.items():
                for i in range(len(pretoken) - 1):
                    pair = (pretoken[i], pretoken[i + 1])
                    byte_pair_freqs[pair] += freq

            # Find most frequent byte pair, merge it and add to vocab.
            most_frequent_pair = max(
                byte_pair_freqs.items(), key=lambda item: item[1])[0]
            token_1 = reverse_vocab[int(most_frequent_pair[0])]
            token_2 = reverse_vocab[int(most_frequent_pair[1])]
            new_token = token_1 + token_2
            new_token_id = len(vocab)
            vocab[new_token] = new_token_id
            reverse_vocab[new_token_id] = new_token
            merges.append((token_1, token_2))

            # Merge byte pairs in pretoken frequencies.
            new_pretoken_freqs: dict[tuple[bytes], int] = defaultdict(int)
            for pretoken, freq in pretoken_freqs.items():
                merged_pretoken = []
                i = 0
                while i < len(pretoken):
                    if (i < len(pretoken) - 1 and
                            (pretoken[i], pretoken[i + 1]) == most_frequent_pair):
                        merged_pretoken.append(new_token_id)
                        i += 2
                    else:
                        merged_pretoken.append(pretoken[i])
                        i += 1
                new_pretoken_freqs[tuple(merged_pretoken)] += freq
            pretoken_freqs = new_pretoken_freqs

        print(f'Tokenizer: Final vocabulary size: {len(vocab)}')
        return vocab, reverse_vocab, merges

    def _save_vocab(self, vocab: dict[bytes, int], merges: list[tuple[bytes, bytes]]) -> None:
        """
        Saves the vocabulary to disk.
        """
        print('Tokenizer: Saving final vocabulary...')
        with open(self.vocab_file_path, 'wb') as file:
            pickle.dump((vocab, merges), file)

    def _flush_pretokens(self) -> None:
        """
        Deletes temporary pretoken files to free up space.
        """
        print('Tokenizer: Cleaning up temporary pretoken files...')
        pretoken_files = list(
            self.pretokens_path_prefix.glob('pretokens_*.pkl'))
        for pretokens_file in pretoken_files:
            os.remove(pretokens_file)
