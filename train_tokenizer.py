from pathlib import Path
from utils import Tokenizer


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
                    500, [b'<|endoftext|>'])
