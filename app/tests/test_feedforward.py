import torch
import unittest

from model.feed_forward import FeedForward


class TestFeedForward(unittest.TestCase):
    def test_feedforward_forward(self):
        d_model = 16
        d_ff = 64
        batch_size = 2
        seq_len = 10
        feed_forward = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = feed_forward(x)

        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertEqual(feed_forward.linear1.weights.shape, (d_ff, d_model))
        self.assertEqual(feed_forward.linear2.weights.shape, (d_ff, d_model))
        self.assertEqual(feed_forward.linear3.weights.shape, (d_model, d_ff))
