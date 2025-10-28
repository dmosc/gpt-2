from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    # max sequence length
    block_size: int = 1024
    # number of tokens: 50k BPE merges + 256-byte tokens + 1 <|endoftext|>
    vocab_size: int = 50257
    # number of layers
    n_layer: int = 12
    # number of attention heads
    n_head: int = 12
    # embedding dimension
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias but a "mask". this was the name used in the original
        # GPT code.
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size,
                                                           config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dimension.
        # nh is "number of heads", hs is "head size", and C (number of channels)
        # which is nh * hs.
        # e.g. in GPT-2 (124M), nh=12, hs=64, nh * hs = 768 = C channels in the
        # transformer.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materializes the large (T, T) matrix for all the queries
        # and keys).
        att = (q @ k.transpose(-2, -1) *
               (1.0 / torch.sqrt(torch.tensor(k.size(-1)))))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # Token embedding weights.
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # Position embedding weights.
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # Hidden layers containing the transformer blocks.
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Layer norm at the end of the transformer.
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying scheme: https://arxiv.org/pdf/1608.05859
        self.transformer.wte.weight = self.lm_head.weight
        # Initialize weights a la GPT-2. .apply will recursively apply the
        # _init_weights method to every module in the model.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T) where B is batch size, T is sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the token and position embeddings
        # (T)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pretrained GPT model."""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        print(f'loading weights from pretrained model: {model_type}')
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            # 1.5B params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask/ buffer from our model's state dict
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        # load a HuggingFace GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all params are aligned and match in names/ shapes
        sd_keys_hf = sd_hf.keys()
        # disard the masked bias from HF model's state dict
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]
        # discard the attention bias from HF model's state dict
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight',
                      'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # the OpenAI checkpoint uses a "conv1d" equivalent weight layout, but
        # we only want to use a vanilla version which means that we have to
        # transpose these weights as we copy them over.
        assert len(sd_keys) == len(
            sd_keys_hf), f'mismatched number of keys {len(sd_keys)} vs {len(sd_keys_hf)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for conv1d weights, we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
