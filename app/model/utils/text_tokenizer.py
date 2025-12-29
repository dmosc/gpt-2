import re
from collections import defaultdict
from pathlib import Path
import pickle
from typing import Optional

from .tokenizer import Tokenizer


class TextTokenizer(Tokenizer):
    # OpenAI's Tiktoken pre-tokenizer regex pattern:
    # https://github.com/openai/tiktoken/pull/234/files
    pretokenizer_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\s\\p{L}\\p{N}]+|\s+(?!\S)|\s+"""
    end_of_chunk_split_token = b'<|endoftext|>'

    def __init__(self):
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
            self.vocab, self.merges = pickle.load(file)

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
    
    def _process_chunk(self, dataset_path: Path, start_offset: int,
                       end_offset: int):
        # Read chunk from dataset and pretokenize.
        pretokens: list[bytes] = []
        with open(dataset_path, 'rb') as file:
            file.seek(start_offset)
            chunk = file.read(end_offset - start_offset)
            pretokens = self._pretokenize_sequence(
                chunk.decode('utf-8', errors='ignore'))

        # Save pretokens to temporary file.
        pretokens_file_path = self.tmp_dir_path / \
            f'pretokens_{start_offset}_{end_offset}.pkl'
        pretokens_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pretokens_file_path, 'wb') as file:
            pickle.dump(pretokens, file)

    def _pretokenize_sequence(self, sequence: str) -> list[bytes]:
        """
        Pretokenizes a single byte sequence using the pretokenizer regex
        pattern.

        Args:
            sequence (str): The string sequence to pretokenize.

        Returns:
            list[bytes]: List of pretokens.
        """
        pattern = re.compile(self.pretokenizer_pattern)
        matches = pattern.finditer(sequence)
        return [match.group().encode('utf-8') for match in matches]

    def _build_vocab(self, max_vocab_size: int, special_tokens: list[bytes],
                     token_freqs: dict[tuple[bytes], int]) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
        """
        Builds the vocabulary from pretokenized files and performs BPE merging.

        Args:
            max_vocab_size (int): Maximum size of the vocabulary.
            special_tokens (list[bytes]): List of special tokens to include in the vocabulary.
        """
        # Initialize vocabulary and reverse vocabulary.
        vocab, reverse_vocab = self._init_vocab(special_tokens)
        # Perform BPE merging to build vocabulary.
        vocab, reverse_vocab, merges = self._bpe_merge(vocab, reverse_vocab,
                                                       token_freqs,
                                                       max_vocab_size)
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
