from pathlib import Path

from model.modules import LanguageModel
from model.modules.optimizers import AdamW
from model.modules.schedulers import CosAnnealingScheduler
from model.modules.utils import Evaluator, grad_clip, DataLoader, Checkpointer
from config import Config


def train_model():
    print('Training model...')
    config = Config()
    model = LanguageModel(config.d_model, config.num_heads, config.d_ff,
                          config.vocab_size, config.max_seq_len,
                          config.num_layers)
    scheduler = CosAnnealingScheduler(config.max_lr, config.min_lr,
                                      config.warmup_steps, config.max_steps)
    optimizer = AdamW(list(model.parameters()),
                      lr=scheduler.max_lr, weight_decay=config.weight_decay)
    base_dir = Path(__file__).resolve().parent
    dataloader = DataLoader(config.tokenizer, base_dir / config.data_path,
                            config.batch_size, config.seq_len)
    checkpointer = Checkpointer(base_dir / config.checkpoint_dir)
    evaluator = Evaluator()

    for epoch in range(config.epochs):
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
            if step % config.save_every_n_steps == 0:
                checkpointer.save_checkpoint(step, model, optimizer, evaluator,
                                             config)
