import torch
import argparse

from pathlib import Path
from wakepy import keep
from trainer import Trainer
from model.config import Config


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        torch.set_default_device('mps')
        print("Using MPS device for training.")
    else:
        print("MPS device not available. Using default device.")
    parser = argparse.ArgumentParser(prog='train_all.py',
                                     description='Train tokenizer and model sequentially.',
                                     epilog='Example usage: python app/train_all.py --skip_tokenizer_training')
    parser.add_argument('--skip_tokenizer_training', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a model checkpoint to resume training from.')
    args = parser.parse_args()
    with keep.presenting():
        data_dir = Path(__file__).parent / 'data'
        checkpoint_path = Path(
            args.checkpoint_path) if args.checkpoint_path else None
        config = Config(data_dir, checkpoint_path)
        trainer = Trainer(config)
        if args.skip_tokenizer_training:
            print("Skipping tokenizer training. Loading from memory.")
        else:
            trainer.train_tokenizer()
        trainer.train_model()
