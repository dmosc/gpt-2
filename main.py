import time
import torch
from torch.nn import functional as F

from train import GPT, GPTConfig
from dataloader import DataLoader

# Global variables.
# 2**19, ~0.5M tokens per batch to process (GPT-2 paper)
gpt_token_batch_size = 524288
# Example batch size to process at once.
batch_size = 16
# Context length / sequence length.
block_size = 1024
assert gpt_token_batch_size % (batch_size * block_size) == 0, \
    'gpt_token_batch_size must be divisible by (batch_size * block_size)'
# Number of sequences to process before stepping the optimizer.
gradient_accumulation_steps = gpt_token_batch_size // (batch_size * block_size)
# Other hyperparameters.
learning_rate = 3e-4
steps = 50

print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    device = get_device()
    print(f'running on device: {device}')
    model = GPT(GPTConfig(vocab_size=50304))
    model.eval()
    model.to(device)
    # This is a default preference that should always be set to improve training
    # speed but I'm disabling it for now because with not enough resources it
    # fails with "Not enough SMs to use max_autotune_gemm mode".
    # model = torch.compile(model)
    print('model loaded successfully')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps)
    data_loader = DataLoader('input.txt', batch_size, block_size)
    for step in range(steps):
        start_time = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for partial_step in range(gradient_accumulation_steps):
            x, y = data_loader.get_next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x.to(device), y.to(device))
            # Scale the loss to account for gradient accumulation. The gradients
            # just add on each successive backward() call. Addition of gradients
            # corresponds to a sum in the objective, but instead of a sum we
            # want mean.
            loss /= gradient_accumulation_steps
            loss.backward()
            loss_accum += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        torch.mps.synchronize()
        end_time = time.time()
        tok_per_sec = (batch_size * block_size *
                       gradient_accumulation_steps) / (end_time - start_time)
        print(
            f'step {step}/{steps}, loss: {loss_accum:.4f}, norm: {norm:.4f}, time: {end_time - start_time:.4f}s, tokens/sec: {tok_per_sec:.2f}')
