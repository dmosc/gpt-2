from tokenizer import Tokenizer
from pathlib import Path


def main() -> None:
    tokenizer = Tokenizer()
    # tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
    #                 max_vocab_size=500,
    #                 special_tokens=[b'<|endoftext|>'])
    # print(tokenizer.vocab)
    tokenizer.load()
    print(tokenizer.vocab)

if __name__ == '__main__':
    main()
