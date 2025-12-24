import argparse

from pathlib import Path
from wakepy import keep
from trainer import Trainer
from model.modules.config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_all.py',
                                     description='Train tokenizer and model sequentially.',
                                     epilog='Example usage: python app/train_all.py --skip_tokenizer_training')
    parser.add_argument('--skip_tokenizer_training', action='store_true')
    args = parser.parse_args()
    with keep.presenting():
        data_dir = Path(__file__).parent / 'data'
        config = Config(data_dir)
        trainer = Trainer(config)
        if args.skip_tokenizer_training:
            print("Skipping tokenizer training. Loading from memory.")
        else:
            trainer.train_tokenizer()
        trainer.train_model()
