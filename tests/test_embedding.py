import torch
import unittest

from parameterized import parameterized

from app.model.embedding import Embedding


class TestEmbedding(unittest.TestCase):
    @parameterized.expand([
        ((100, 128), (10, 1024)),
        ((100, 128), (10, 10, 1024)),
        ((100, 128), (10, 10, 10, 1024)),
    ])
    def test_embedding_forward(self, embedding_dims: tuple, input_dims: tuple):
        num_embeddings, embedding_dim = embedding_dims
        embedding = Embedding(num_embeddings, embedding_dim)
        x = torch.randint(low=0, high=num_embeddings, size=input_dims)
        output = embedding(x)

        assert embedding.embeddings.shape == (num_embeddings, embedding_dim)
        assert type(embedding.embeddings) == torch.nn.Parameter
        assert embedding.embeddings.requires_grad is True
        self.assertAlmostEqual(embedding.embeddings.std().item(), 1, delta=0.1)
        self.assertTrue(torch.all(output >= -3))
        self.assertTrue(torch.all(output <= 3))
        assert output.shape == (*input_dims, embedding_dim)
