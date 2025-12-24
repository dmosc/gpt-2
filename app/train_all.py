from pathlib import Path
from wakepy import keep
from trainer import Trainer
from model.modules.config import Config


if __name__ == '__main__':
    with keep.presenting():
        data_dir = Path(__file__).parent / 'data'
        config = Config(data_dir)
        trainer = Trainer(config)
        trainer.train_tokenizer()
        trainer.train_model()
