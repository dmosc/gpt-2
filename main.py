import torch

from tokenizer import Tokenizer
from pathlib import Path
from modules import LanguageModel


def main() -> None:
    tokenizer = Tokenizer()
    tokenizer.load()
    # tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
    #                 max_vocab_size=500,
    #                 special_tokens=[b'<|endoftext|>'])
    # print(tokenizer.vocab, tokenizer.merges)
    d_model = 128
    num_heads = 8
    d_ff = int(8 / 3 * d_model)
    vocab_size = len(tokenizer.vocab)
    max_seq_len = 1024
    num_layers = 4
    lm = LanguageModel(d_model, num_heads, d_ff, vocab_size, max_seq_len,
                       num_layers)
    batch = torch.tensor([
        tokenizer.encode('what\'s your '),
        tokenizer.encode('my favorite ice cream is'),
    ])
    outputs = lm.generate_next_token(batch).tolist()
    for idx in range(len(batch)):
        decoded_seq = tokenizer.decode(batch[idx].tolist())
        next_word = tokenizer.decode([outputs[idx]])
        print(f'{decoded_seq} -> {next_word}')


if __name__ == '__main__':
    main()
