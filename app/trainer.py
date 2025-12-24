from model.modules.language_model import LanguageModel
from model.modules.config import Config
from model.modules.optimizers.adamw import AdamW
from model.modules.schedulers.cos_annealing import CosAnnealingScheduler
from model.modules.utils.evaluator import Evaluator
from model.modules.utils.grad_clip import grad_clip
from model.modules.utils.dataloader import DataLoader
from model.modules.utils.checkpointer import Checkpointer
from model.modules.utils.tokenizer import Tokenizer


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train_model(self):
        print('Training model...')
        tokenizer = Tokenizer()
        tokenizer.load()
        model = LanguageModel(self.config)
        scheduler = CosAnnealingScheduler(self.config)
        optimizer = AdamW(list(model.parameters()), self.config)
        dataloader = DataLoader(self.config, tokenizer)
        evaluator = Evaluator()

        for epoch in range(self.config.epochs):
            step = 0
            print(f'{epoch=}')
            while data_paylaod := dataloader.get_next_batch():
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
                grad_clip(model.parameters(), max_norm=0.1)
                optimizer.step()
                step += 1
                if step % self.config.save_every_n_steps == 0:
                    Checkpointer.save_checkpoint(step, model, optimizer,
                                                 evaluator, self.config)

    def train_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.train(self.config.valid_data_path,
                                    self.config.max_vocab_size,
                                    [b'<|endoftext|>'])
