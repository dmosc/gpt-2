import torch

from pathlib import Path
from .evaluator import Evaluator

from ..config import Config
from ..language_model import LanguageModel
from ..optimizers.adamw import AdamW


class Checkpointer:
    @staticmethod
    def save_checkpoint(step: int, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, evaluator: Evaluator,
                        config: Config):
        path = config.get_checkpoint_path(step)
        if path.exists():
            print(f'{path} already exists; overwriting it...')
        state = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.state_dict(),
            'evaluator': evaluator.state_dict(),
        }
        torch.save(state, path)
        print(f'Saved checkpoint at step {step} to {path}')

    @staticmethod
    def load_checkpoint(path: Path) -> tuple[LanguageModel, AdamW, Evaluator, Config]:
        if not path.exists():
            raise FileNotFoundError(
                f'Checkpoint at path={path} doesn\'t exist.')
        torch.serialization.add_safe_globals([Path])
        state = torch.load(path, weights_only=False)
        config = Config.load_state_dict(state['config'])
        evaluator = Evaluator.load_state_dict(state['evaluator'])
        model = LanguageModel(config)
        model.load_state_dict(state['model'])
        optimizer = AdamW(list(model.parameters()), config)
        optimizer.load_state_dict(state['optimizer'])
        print(f'Loaded checkpoint {path}')
        return model, optimizer, evaluator, config
