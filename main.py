import torch
import matplotlib.pyplot as plt

from tokenizer import Tokenizer
from pathlib import Path

from modules import LanguageModel
from modules.optimizers import SGD, AdamW
from modules.schedulers import CosAnnealingScheduler
from utils import cross_entropy, perplexity, grad_clip


def main() -> None:
    tokenizer = Tokenizer()
    tokenizer.load()
    # tokenizer.train(Path('data/TinyStoriesV2-GPT4-valid.txt'),
    #                 max_vocab_size=500,
    #                 special_tokens=[b'<|endoftext|>'])
    # print(tokenizer.vocab, tokenizer.merges)
    d_model = 128
    num_heads = 8
    d_ff = int(8 / 3 * d_model)
    vocab_size = len(tokenizer.vocab)
    max_seq_len = 1024
    num_layers = 4
    lm = LanguageModel(d_model, num_heads, d_ff, vocab_size, max_seq_len,
                       num_layers)
    batch = torch.tensor([
        tokenizer.encode('what\'s your '),
        tokenizer.encode('my favorite ice cream is'),
    ])
    outputs = lm.generate_next_token(batch)
    for idx in range(len(batch)):
        decoded_seq = tokenizer.decode(batch[idx].tolist())
        next_word = tokenizer.decode([outputs[idx].item()])
        print(f'{decoded_seq} -> {next_word}')

    logits = lm._compute_logits(batch)
    targets = torch.randint(0, vocab_size, logits.shape[:-1])
    loss = cross_entropy(logits, targets)
    print(f'{loss.item()=}, {perplexity(loss).item()=}')

    weights = torch.nn.Parameter(5 * torch.rand((10, 10)))
    optim = SGD([weights], lr=1)
    for t in range(100):
        optim.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        optim.step()

    weights_adamw = torch.nn.Parameter(5 * torch.rand((10, 10)))
    optim_adamw = AdamW([weights_adamw], lr=0.01,
                        betas=(0.9, 0.999), weight_decay=0.01)

    for t in range(100):
        optim_adamw.zero_grad()
        loss = (weights_adamw ** 2).mean()
        if t % 10 == 0:
            print(f"Step {t}: loss = {loss.cpu().item():.6f}")
        loss.backward()
        optim_adamw.step()

    scheduler = CosAnnealingScheduler(
        max_lr=0.1,        # Maximum learning rate
        min_lr=0.001,      # Minimum learning rate
        warmup_steps=20,   # Warm-up for 20 steps
        max_steps=100      # Total 100 steps
    )
    weights = torch.nn.Parameter(torch.rand((10, 10)))
    optimizer = AdamW([weights], lr=scheduler.max_lr)

    for step in range(120):
        # Update learning rate
        lr = scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step
        optimizer.zero_grad()
        loss = (weights ** 2).mean()

        if step % 20 == 0:
            print(f"Step {step}: lr = {lr:.6f}, loss = {loss.item():.6f}")

        loss.backward()
        optimizer.step()

    params = [
        torch.nn.Parameter(torch.ones(10, 10)),
        torch.nn.Parameter(torch.ones(5, 5))
    ]

    # Create artificially large gradients
    params[0].grad = torch.randn(10, 10) * 100  # Large gradients
    params[1].grad = torch.randn(5, 5) * 50

    # Check norm before clipping
    original_norm = torch.sqrt(
        sum(p.grad.data.norm(2) ** 2 for p in params)
    ).item()
    print(f"Original gradient norm: {original_norm:.4f}")

    # Apply gradient clipping
    max_norm = 1.0
    returned_norm = grad_clip(params, max_norm=max_norm)
    print(f"Returned norm (should match original): {returned_norm:.4f}")

    # Check norm after clipping
    clipped_norm = torch.sqrt(
        sum(p.grad.data.norm(2) ** 2 for p in params)
    ).item()
    print(f"Clipped gradient norm: {clipped_norm:.4f}")
    print(
        f"Should be close to max_norm ({max_norm}): {abs(clipped_norm - max_norm) < 1e-5}")

    # Integration with training loop
    print("\n--- Integration with optimizer ---")
    model = torch.nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=0.01)

    for step in range(5):
        optimizer.zero_grad()

        # Forward pass
        x = torch.randn(32, 10)
        loss = model(x).sum()
        loss.backward()

        # Gradient clipping before optimizer step
        grad_norm = grad_clip(model.parameters(), max_norm=1.0)
        print(f"Step {step}: grad_norm = {grad_norm:.4f}")

        optimizer.step()


if __name__ == '__main__':
    main()
