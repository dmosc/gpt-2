from typing import Optional
from pathlib import Path
from numerize import numerize


class Config:
    def __init__(self, data_dir: Path, checkpoint_path: Optional[Path] = None):
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = int(8 / 3 * self.d_model)
        self.max_seq_len = 1024
        self.num_layers = 4
        self.max_lr = 3e-2
        self.min_lr = 3e-3
        self.warmup_steps = 1000
        self.max_steps = 3000
        self.weight_decay = 0.1
        self.batch_size = 16
        self.seq_len = 1024
        self.save_every_n_steps = 100
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.train_data_path = self.data_dir / 'TinyStoriesV2-GPT4-train.txt'
        self.valid_data_path = self.data_dir / 'TinyStoriesV2-GPT4-valid.txt'
        self.checkpoint_dir = self.data_dir / 'models'
        self.state_file = Path('state.pkl')
        self.epochs = 10
        self.vocab_size = 5000
        self.lr = 3e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.special_tokens = [b'<|endoftext|>']

    @staticmethod
    def load_state_dict(state_dict: dict) -> 'Config':
        config = Config(state_dict['data_dir'])
        config.__dict__.update(state_dict)
        return config

    def state_dict(self):
        return self.__dict__

    def get_checkpoint_path(self, num_params: int, step: int) -> Path:
        model_specs = f'{numerize.numerize(num_params)}_{numerize.numerize(self.seq_len)}'
        path = self.checkpoint_dir / model_specs / str(step) / self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
