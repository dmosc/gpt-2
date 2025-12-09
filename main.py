from tokenizer import Tokenizer
from pathlib import Path


def main() -> None:
    tokenizer = Tokenizer()
    tokenizer.load()
    # tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
    #                 max_vocab_size=500,
    #                 special_tokens=[b'<|endoftext|>'])
    # print(tokenizer.vocab, tokenizer.merges)
    tokens = tokenizer.encode('could you help me right now?')
    print(tokens)
    print(tokenizer.decode(tokens))


if __name__ == '__main__':
    main()
