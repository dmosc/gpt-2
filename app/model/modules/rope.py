import torch


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_qk: int, max_seq_len: int, device=None,
                 dtype=torch.float32):
        super().__init__()
        assert d_qk % 2 == 0, 'd_qk must be even for RoPE'
        self.theta = theta
        self.d_qk = d_qk
        # 1. Create a range for 'i' positions and 'k' pairs.
        # Note that positions: [[0], [1], [...], [max_seq_len - 1]] should be
        # shaped as a column tensor so that broadcasting can occur when
        # computing the actual angle rotations.
        positions = torch.arange(
            max_seq_len, device=device, dtype=dtype).reshape(max_seq_len, 1)
        # pair_indices: [1, 2, ..., d_qk / 2] starting at "1" to keep the
        # "2k - 2" expression implementation without extra logic to avoid
        # negative values.
        pair_indicies = torch.arange(1, d_qk // 2 + 1, device=device,
                                     dtype=dtype)
        # 2. Compute the angles theta_{i, k} = i / (theta ** ((2k - 2) / d_qk))
        angles = positions / (theta ** (2 * pair_indicies - 2) / d_qk)
        # 3. Precompute and register buffers (non-learnable params). We repeat
        # each angle twice so they match the shape of the d_qk vector in the
        # forward call.
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_qk, f'Input tensor dimension {x.shape[-1]=} must match {self.d_qk=}'
        # 1. Extract the precomputed angles for the tokens in the sequence.
        cos = self.cos[tuple(token_positions)]
        sin = self.sin[tuple(token_positions)]
        # 2. Vectorize the rotation by stacking odd indices negated and even
        # indices. The resulting shape after stacking is (...,  d_qk // 2, 2)
        # so we have to flatten the last two dimension together to match the
        # input's last dimension.
        x_rotated = torch.stack(
            [-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return (x * cos) + (x_rotated * sin)
