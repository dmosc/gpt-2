import os
import re
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
import pickle


class Tokenizer:
    default_chunk_size_bytes = 100_000  # 100 KB
    default_parallel_processes = os.cpu_count()
    # OpenAI's Tiktoken pre-tokenizer regex pattern:
    # https://github.com/openai/tiktoken/pull/234/files
    pretokenizer_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\s\\p{L}\\p{N}]+|\s+(?!\S)|\s+"""
    pretokens_path_prefix = Path('/tmp/gpt-2')
    vocab_file_path = pretokens_path_prefix / 'vocab.pkl'
    end_of_chunk_split_token = b'<|endoftext|>'

    def __init__(self, dataset_path: Path, max_vocab_size: int,
                 special_tokens: list[bytes]) -> None:
        self.dataset_path = dataset_path
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.vocab.update(
            {token: 256 + i for i, token in enumerate(special_tokens)})
        self._reverse_vocab: dict[int, bytes] = {
            v: k for k, v in self.vocab.items()}

    def train(self) -> None:
        chunk_offsets = self._get_dataset_chunk_offsets()
        iterable_args = zip(chunk_offsets[:-1], chunk_offsets[1:])
        print(f'Tokenizer: Pre-tokenizing {len(chunk_offsets) - 1} chunks '
              f'using {self.default_parallel_processes} parallel processes...')
        with Pool(processes=self.default_parallel_processes) as pool:
            pool.starmap(self._pretokenize_file_chunk, iterable_args)
        print('Tokenizer: Building vocabulary...')
        self._build_vocab()
        print('Tokenizer: Cleaning up temporary pretoken files...')
        self._flush_pretokens()

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
        pretokens_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pretokens_file_path, 'wb') as file:
            pickle.dump(pretokens, file)

    def _build_vocab(self) -> None:
        """
        Builds the vocabulary from pretokenized files and performs BPE merging.
        """
        # Load pretoken and compute frequencies across all files.
        pretoken_files = list(
            self.pretokens_path_prefix.glob('pretokens_*.pkl'))
        pretoken_freqs: dict[tuple[bytes], int] = defaultdict(int)
        for pretokens_file in pretoken_files:
            with open(pretokens_file, 'rb') as file:
                for pretoken in pickle.load(file):
                    pretoken_freqs[tuple(pretoken)] += 1

        # Perform BPE merging to build vocabulary.
        self._bpe_merge(pretoken_freqs)

        # Save final vocabulary to file.
        print('Tokenizer: Saving final vocabulary...')
        with open(self.pretokens_path_prefix / 'vocab.pkl', 'wb') as file:
            pickle.dump(self.vocab, file)

    def _bpe_merge(self, pretoken_freqs: dict[tuple[bytes], int]) -> None:
        """
        Performs BPE merging iteratively until max_vocab_size is reached.
        Optimizes frequency calculation by only updating where merges occur.
        """
        while len(self.vocab) < self.max_vocab_size:
            # Compute byte pair frequencies.
            byte_pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
            for pretoken, freq in pretoken_freqs.items():
                for i in range(len(pretoken) - 1):
                    pair = (pretoken[i], pretoken[i + 1])
                    byte_pair_freqs[pair] += freq

            # Find most frequent byte pair, merge it and add to vocab.
            most_frequent_pair = max(
                byte_pair_freqs.items(), key=lambda item: item[1])[0]
            new_token = self._reverse_vocab[int(most_frequent_pair[0])] + \
                self._reverse_vocab[int(most_frequent_pair[1])]
            new_token_id = len(self.vocab)
            self.vocab[new_token] = new_token_id
            self._reverse_vocab[new_token_id] = new_token

            # Merge byte pairs in pretoken frequencies
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

        print(f'Tokenizer: Final vocabulary size: {len(self.vocab)}')

    def _flush_pretokens(self) -> None:
        """
        Deletes temporary pretoken files to free up space.
        """
        pretoken_files = list(
            self.pretokens_path_prefix.glob('pretokens_*.pkl'))
        for pretokens_file in pretoken_files:
            os.remove(pretokens_file)
