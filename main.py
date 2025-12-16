import torch

from pathlib import Path

from modules import LanguageModel
from modules.optimizers import AdamW
from modules.schedulers import CosAnnealingScheduler
from utils import cross_entropy, perplexity, grad_clip, Tokenizer, DataLoader


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

    max_lr = 0.1
    min_lr = 0.001
    warmup_steps = 20
    max_steps = 100
    scheduler = CosAnnealingScheduler(max_lr, min_lr, warmup_steps, max_steps)
    optimizer = AdamW(list(model.parameters()), lr=scheduler.max_lr)

    batch_size = 32
    seq_len = 1024
    dataloader = DataLoader(tokenizer,
                            Path('data/TinyStoriesV2-GPT4-valid.txt'),
                            batch_size, seq_len)

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
            # Performs model(batch) under the hood but extracts the last column
            # to get the predictions for each of the sequences and compare
            # against targets.
            logits = model.generate_next_token(batch)
            loss = cross_entropy(logits, targets)
            print(f'{loss.item()=}, {perplexity(loss).item()=}')
            loss.backward()
            optimizer.step()
            step += 1


if __name__ == '__main__':
    main()
