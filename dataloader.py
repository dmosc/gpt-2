import tiktoken
import torch


class DataLoader:
    def __init__(self, file_path, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokenizer = tiktoken.get_encoding('gpt2')
        with open(file_path, 'r') as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.pointer = 0

    def get_next_batch(self):
        if self.pointer + self.batch_size * self.block_size + 1 > len(self.data):
            self.pointer = 0
        x = self.data[self.pointer: self.pointer +
                      self.batch_size * self.block_size]
        y = self.data[self.pointer + 1: self.pointer +
                      1 + self.batch_size * self.block_size]
        self.pointer += self.batch_size * self.block_size
        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)
        return x, y
