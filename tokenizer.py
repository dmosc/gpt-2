import os
import re
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
import pickle
import random


class Tokenizer:
    default_chunk_size_bytes = 100_000  # 100 KB
    # OpenAI's Tiktoken pre-tokenizer regex pattern:
    # https://github.com/openai/tiktoken/pull/234/files
    pretokenizer_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\s\\p{L}\\p{N}]+|\s+(?!\S)|\s+"""
    pretokens_path_prefix = Path('/tmp/gpt-2')
    end_of_chunk_split_token = b'<|endoftext|>'

    def __init__(self, dataset_path: Path, max_vocab_size: int,
                 special_tokens: list[bytes]) -> None:
        self.dataset_path = dataset_path
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[bytes, int] = {}

    def train(self) -> None:
        chunk_offsets = self._get_dataset_chunk_offsets()
        iterable_args = zip(chunk_offsets[:-1], chunk_offsets[1:])
        with Pool() as pool:
            pool.starmap(self._pretokenize_file_chunk, iterable_args)

    def _get_dataset_chunk_offsets(self) -> list[int]:
        with open(self.dataset_path, 'rb') as file:
            dataset_size_bytes = os.path.getsize(self.dataset_path)
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
            with open(self.dataset_path, 'rb') as file:
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

    def _pretokenize_file_chunk(self, start_offset: int, end_offset: int) -> None:
        # Read chunk from dataset and pretokenize
        pretokens: list[bytes] = []
        with open(self.dataset_path, 'rb') as file:
            file.seek(start_offset)
            chunk = file.read(end_offset - start_offset)
            pattern = re.compile(self.pretokenizer_pattern, re.UNICODE)
            matches = pattern.findall(chunk.decode('utf-8', errors='ignore'))
            pretokens = [match.encode('utf-8') for match in matches]

        # Save pretokens to temporary file
        pretokens_file_path = self.pretokens_path_prefix / \
            f'pretokens_{start_offset}_{end_offset}.pkl'
        print(f'Saving pretokens to {pretokens_file_path}')
        pretokens_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pretokens_file_path, 'wb') as file:
            pickle.dump(pretokens, file)
