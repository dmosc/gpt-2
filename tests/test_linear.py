import torch
import unittest

from parameterized import parameterized

from app.model.linear import Linear


class TestLinear(unittest.TestCase):
    @parameterized.expand([
        ((128, 384), (10, 128)),
        ((128, 384), (10, 10, 128)),
        ((128, 384), (10, 10, 10, 128)),
    ])
    def test_linear_forward(self, linear_dims: tuple, input_dims: tuple):
        linear = Linear(linear_dims[0], linear_dims[1])
        x = torch.rand(size=input_dims)
        output = linear(x)

        assert linear.weights.shape == (linear_dims[1], linear_dims[0])
        assert type(linear.weights) == torch.nn.Parameter
        assert linear.weights.requires_grad is True
        self.assertAlmostEqual(linear.weights.std().item(), linear.std,
                               delta=0.2)
        self.assertTrue(torch.all(output >= -3 * linear.std ** 0.5))
        self.assertTrue(torch.all(output <= 3 * linear.std ** 0.5))
        assert output.shape == (*input_dims[:-1], linear_dims[1])


if __name__ == '__main__':
    unittest.main()
