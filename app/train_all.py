from wakepy import keep
from trainer import Trainer
from model.modules import Config


if __name__ == '__main__':
    with keep.presenting():
        config = Config()
        trainer = Trainer(config)
        trainer.train_tokenizer()
        trainer.train_model()
