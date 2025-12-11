import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embeddings = torch.rand(size=(num_embeddings, embedding_dim),
                                dtype=dtype)
        embeddings = torch.nn.init.trunc_normal_(embeddings, mean=0, std=1,
                                                 a=-3, b=3)
        self.embeddings = torch.nn.Parameter(embeddings)
        self.device = device
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embeddings[tokens]
