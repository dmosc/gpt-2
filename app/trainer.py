from model.language_model import LanguageModel
from model.config import Config
from model.optimizers.adamw import AdamW
from model.schedulers.cos_annealing import CosAnnealingScheduler
from model.utils.evaluator import Evaluator
from model.utils.grad_clip import grad_clip
from model.utils.dataloader import DataLoader
from model.utils.checkpointer import Checkpointer
from model.utils.tokenizer import Tokenizer


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train_model(self):
        print('Training model...')
        dataloader, model, optimizer, scheduler, evaluator, step = self._unpack_training_components()
        for epoch in range(self.config.epochs):
            print(f'{epoch=}')
            while data_paylaod := dataloader.get_next_batch(randomize=True):
                batch, targets = data_paylaod
                # Update learning rates following a cosine annealing schedule.
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # Compute gradients, loss and perform a step.
                optimizer.zero_grad()
                logits = model(batch)
                # Reshape for cross entropy
                # (batch_size * seq_len, vocab_size)
                logits = logits.reshape(-1, logits.size(-1))
                # (batch_size * seq_len,)
                targets = targets.reshape(-1)
                loss = evaluator.evaluate(step, logits, targets)
                loss.backward()
                grad_clip(model.parameters(), max_norm=self.config.max_norm)
                optimizer.step()
                step += 1
                if step % self.config.save_every_n_steps == 0:
                    Checkpointer.save_checkpoint(step, model, optimizer,
                                                 evaluator, self.config)

    def train_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.train(self.config.valid_data_path, self.config.vocab_size,
                        self.config.special_tokens)

    def _unpack_training_components(self) -> tuple[DataLoader, LanguageModel, AdamW, CosAnnealingScheduler, Evaluator, int]:
        tokenizer = Tokenizer()
        tokenizer.load()
        dataloader = DataLoader(self.config, tokenizer)
        scheduler = CosAnnealingScheduler(self.config)

        if self.config.checkpoint_path:
            print(
                f'Resuming training from checkpoint: {self.config.checkpoint_path}')
            model, optimizer, evaluator, _ = Checkpointer.load_checkpoint(
                self.config.checkpoint_path)
            # Second to last value in the .parts attribute of the path is the
            # step number the checkpoint was saved at. We recoup it like this
            # and use it to sync the initial step number with the loaded
            # checkpoint.
            step = int(self.config.checkpoint_path.parts[-2])
            return dataloader, model, optimizer, scheduler, evaluator, step
        else:
            model = LanguageModel(self.config)
            optimizer = AdamW(list(model.parameters()), self.config)
            evaluator = Evaluator()
            return dataloader, model, optimizer, scheduler, evaluator, 0
