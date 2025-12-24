from pathlib import Path
from utils import Tokenizer


class Config:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = int(8 / 3 * self.d_model)
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_seq_len = 1024
        self.num_layers = 4
        self.max_lr = 3e-3
        self.min_lr = 2e-3
        self.warmup_steps = 100
        self.max_steps = 2000
        self.weight_decay = 0.1
        self.batch_size = 16
        self.seq_len = 1024
        self.save_every_n_steps = 2000
        self.data_path = Path('data/TinyStoriesV2-GPT4-train.txt')
        self.valid_data_path = Path('data/TinyStoriesV2-GPT4-valid.txt')
        self.checkpoint_dir = Path('data/models')
        self.epochs = 100

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
