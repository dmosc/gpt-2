import tiktoken
import torch
from torch.nn import functional as F

from train import GPT, GPTConfig


if __name__ == '__main__':
    # detect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f'device: {device}')
    num_return_sequences = 5
    max_length = 30
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    print('model loaded successfully')
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode("Once upon a time")
    # (T,)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    # (num_return_sequences, T)
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    # generate
    torch.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            # (B, T, vocab_size)
            logits = model(x)
            # take the logits at the last position
            # (B, vocab_size)
            logits = logits[:, -1, :]
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (HF pipeline default)
            # topk_probs and topk_indices becomes (num_return_sequences, 50)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            # select a token from the top-k probs
            # (B, 1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            # gather corresponding token indices
            # (B, 1)
            xcol = torch.gather(topk_indices, dim=-1, index=ix)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)
    # decode the generated sequences
    for i in range(num_return_sequences):
        generated_sequence = x[i].tolist()
        text = tokenizer.decode(generated_sequence)
        print(f'=== GENERATED SEQUENCE {i+1} ===')
        print(text)
        print()
