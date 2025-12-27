import torch
import unittest

from unittest.mock import MagicMock

from app.model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def test_forward(self):
        d_model = 32
        num_heads = 4
        d_ff = 64
        max_seq_len = 10
        batch_size = 2
        transformer = Transformer(d_model, num_heads, d_ff, max_seq_len)
        transformer.causal_self_attn_prenorm.forward = MagicMock(
            side_effect=transformer.causal_self_attn_prenorm.forward)
        transformer.causal_self_attn.forward = MagicMock(
            side_effect=transformer.causal_self_attn.forward)
        transformer.feed_forward_prenorm.forward = MagicMock(
            side_effect=transformer.feed_forward_prenorm.forward)
        transformer.feed_forward.forward = MagicMock(
            side_effect=transformer.feed_forward.forward)
        x = torch.randn(batch_size, max_seq_len, d_model)
        output = transformer(x)

        self.assertEqual(output.shape, (batch_size, max_seq_len, d_model))
        transformer.causal_self_attn_prenorm.forward.assert_called_once_with(x)
        transformer.causal_self_attn.forward.assert_called_once()
        transformer.feed_forward_prenorm.forward.assert_called_once()
        transformer.feed_forward.forward.assert_called_once()
