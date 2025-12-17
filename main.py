import torch

from pathlib import Path

from modules import LanguageModel
from modules.optimizers import AdamW
from modules.schedulers import CosAnnealingScheduler
from utils import cross_entropy, perplexity, grad_clip, Tokenizer, DataLoader, Checkpointer


def main() -> None:
    torch.autograd.set_detect_anomaly(True)
    tokenizer = Tokenizer()
    tokenizer.load()
    # tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
    #                 max_vocab_size=500,
    #                 special_tokens=[b'<|endoftext|>'])
    d_model = 128
    num_heads = 8
    d_ff = int(8 / 3 * d_model)
    vocab_size = len(tokenizer.vocab)
    max_seq_len = 1024
    num_layers = 4
    model = LanguageModel(d_model, num_heads, d_ff, vocab_size, max_seq_len,
                          num_layers)

    max_lr = 3e-5
    min_lr = 1e-7
    warmup_steps = 20
    max_steps = 100
    scheduler = CosAnnealingScheduler(max_lr, min_lr, warmup_steps, max_steps)
    optimizer = AdamW(list(model.parameters()),
                      lr=scheduler.max_lr, weight_decay=0.1)

    batch_size = 16
    seq_len = 128
    dataloader = DataLoader(tokenizer,
                            Path('data/TinyStoriesV2-GPT4-valid.txt'),
                            batch_size, seq_len)

    save_every_n_steps = 5_000
    checkpointer = Checkpointer(Path('data/models'))
    epochs = 10
    for epoch in range(epochs):
        step = 0
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
            loss = cross_entropy(logits, targets)
            print(f'{loss.item()=}, {perplexity(loss).item()=}')
            loss.backward()
            grad_clip(model.parameters(), max_norm=0.1)
            optimizer.step()
            step += 1
            if step % save_every_n_steps == 0:
                checkpointer.save_checkpoint(model, optimizer, step)


if __name__ == '__main__':
    main()
