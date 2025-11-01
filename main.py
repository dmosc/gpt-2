import time
import torch
from torch.nn import functional as F

from train import GPT, GPTConfig
from dataloader import DataLoader

# Global variables.
batch_size = 16
block_size = 1024
learning_rate = 3e-4
steps = 50


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
        x, y = data_loader.get_next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x.to(device), y.to(device))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        torch.mps.synchronize()
        end_time = time.time()
        tok_per_sec = batch_size * block_size / (end_time - start_time)
        print(
            f'step {step}/{steps}, loss: {loss.item():.4f}, norm: {norm:.4f}, time: {end_time - start_time:.4f}s, tokens/sec: {tok_per_sec:.2f}')
