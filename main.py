from tokenizer import Tokenizer
from pathlib import Path


def main() -> None:
    dataset_path = Path('data/TinyStoriesV2-GPT4-valid.txt')
    tokenizer = Tokenizer(
        dataset_path=dataset_path,
        max_vocab_size=500,
        special_tokens=[b'<|endoftext|>']
    )
    tokenizer.train()
    print(tokenizer.vocab)


if __name__ == '__main__':
    main()
