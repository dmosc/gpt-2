import torch
from pathlib import Path
from .evaluator import Evaluator

from ..config import Config


class Checkpointer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state_file = 'state.pkl'

    def save_checkpoint(self, step: int, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, evaluator: Evaluator,
                        config: Config):
        output_path = self._get_checkpoint_path(step)
        if output_path.exists():
            print(f'{output_path} already exists; overwriting it...')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.state_dict(),
            'evaluator': evaluator.state_dict(),
            'step': step,
        }
        torch.save(state, output_path)
        print(f'Saved checkpoint at step {step} to {output_path}')

    def load_checkpoint(self, step: int, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, evaluator: Evaluator,
                        config: Config) -> int:
        checkpoint_path = self._get_checkpoint_path(step)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint at step={step} doesn\'t exist.')
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        evaluator.load_state_dict(state['evaluator'])
        config.load_state_dict(state['config'])
        print(f'Loaded checkpoint {checkpoint_path}')
        return state['step']

    def _get_checkpoint_path(self, step: int) -> Path:
        path = self.config.checkpoint_dir / str(step) / self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
