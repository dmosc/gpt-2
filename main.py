import torch

from tokenizer import Tokenizer
from pathlib import Path
from modules import LanguageModel

from utils import cross_entropy, perplexity


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
    outputs = lm.generate_next_token(batch)
    for idx in range(len(batch)):
        decoded_seq = tokenizer.decode(batch[idx].tolist())
        next_word = tokenizer.decode([outputs[idx].item()])
        print(f'{decoded_seq} -> {next_word}')

    logits = lm._compute_logits(batch)
    targets = torch.randint(0, vocab_size, logits.shape[:-1])
    loss = cross_entropy(logits, targets)
    print(f'{loss.item()=}, {perplexity(loss).item()=}')

if __name__ == '__main__':
    main()
