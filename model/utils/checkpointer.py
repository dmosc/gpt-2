import torch
from pathlib import Path
from utils import Evaluator


class Checkpointer:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.state_file = 'state.pkl'

    def save_checkpoint(self, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, evaluator: Evaluator,
                        step: int):
        output_path = self._get_checkpoint_path(step) / self.state_file
        if output_path.exists():
            print(f'{output_path} already exists; overwriting it...')
        state = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'evaluator': evaluator.state_dict(),
        }
        torch.save(state, output_path)
        print(f'Saved checkpoint at step {step} to {output_path}')

    def load_checkpoint(self, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, step: int) -> int:
        checkpoint_path = self._get_checkpoint_path(step) / self.state_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint at step={step} doesn\'t exist.')
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        return state['step']

    def _get_checkpoint_path(self, step: int) -> Path:
        path = self.out_dir / str(step)
        path.mkdir(parents=True, exist_ok=True)
        return path
