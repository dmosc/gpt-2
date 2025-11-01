import time
import torch
from torch.nn import functional as F

from train import GPT, GPTConfig
from dataloader import DataLoader

# Global variables.
batch_size = 16
block_size = 1024
learning_rate = 3e-4
epochs = 50


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
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    print('model loaded successfully')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data_loader = DataLoader('input.txt', batch_size, block_size)
    for epoch in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        x, y = data_loader.get_next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x.to(device), y.to(device))
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        end_time = time.time()
        tok_per_sec = batch_size * block_size / (end_time - start_time)
        print(
            f'epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}, time: {end_time - start_time:.4f}s, tokens/sec: {tok_per_sec:.2f}')
    # while x.size(1) < max_length:
    #     # forward the model to get the logits
    #     with torch.no_grad():
    #         # (B, T, vocab_size)
    #         # take the logits at the last position
    #         # (B, vocab_size)
    #         logits = logits[:, -1, :]
    #         # get the probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # do top-k sampling of 50 (HF pipeline default)
    #         # topk_probs and topk_indices becomes (num_return_sequences, 50)
    #         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
    #         # select a token from the top-k probs
    #         # (B, 1)
    #         ix = torch.multinomial(topk_probs, num_samples=1)
    #         # gather corresponding token indices
    #         # (B, 1)
    #         xcol = torch.gather(topk_indices, dim=-1, index=ix)
    #         # append to the sequence
    #         x = torch.cat((x, xcol), dim=1)
    # decode the generated sequences
    # for i in range(num_return_sequences):
    #     generated_sequence = x[i].tolist()
    #     text = tokenizer.decode(generated_sequence)
    #     print(f'=== GENERATED SEQUENCE {i+1} ===')
    #     print(text)
    #     print()
