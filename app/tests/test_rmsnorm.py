import torch
import unittest

from parameterized import parameterized

from model.rmsnorm import RMSNorm


class TestRMSNorm(unittest.TestCase):
    @parameterized.expand([
        128
    ])
    def test_rmsnorm(self, d_model: int):
        rmsnorm = RMSNorm(d_model)
        x = torch.rand(size=(d_model,))
        output = rmsnorm(x)

        assert rmsnorm.gain.shape == (d_model,)
        assert type(rmsnorm.gain) == torch.nn.Parameter
        assert rmsnorm.gain.requires_grad is True
        assert torch.all(rmsnorm.gain == 1)
        assert output.shape == (d_model,)
        assert torch.allclose(output, torch.ones_like(output), rtol=1)