from pathlib import Path
from model.modules.utils import Tokenizer
from config import Config


def train_tokenizer():
    print('Training tokenizer...')
    config = Config()
    base_dir = Path(__file__).resolve().parent
    tokenizer = Tokenizer()
    tokenizer.train(base_dir / config.valid_data_path, config.max_vocab_size,
                    [b'<|endoftext|>'])
