"""One decode step of a small decoder model, end to end: two layers of a LLaMA-style block.

The class is the whole decoder: every kind of launch a decode step makes, in order -- the embedding gather,
RMSNorm, the q/k/v projections of one token, rotary embedding, grouped-query attention against the cached
keys and values, the output projection, the gated MLP (SiLU), and the vocabulary head. A change to one
kernel shows here at its share of the step; launch overheads and the hand-offs between launches show only
here. The sizes are small (about 0.6 million device cycles on the base: the CA Model runs it in under a
minute, the RTL simulator in hours); a model's own are far larger, where the projections' weights and the cache
stream from memory (`stream_weights_large.py`, `long_context_decode.py`).

Judged at the model tolerance (MODEL = True).
"""
import math

import torch
import torch.nn.functional as F

MODEL = True
DIM, LAYERS, HEADS, KV_HEADS, FFN, VOCAB, PAST = 96, 2, 3, 1, 256, 320, 64
BATCH = 1
HD = DIM // HEADS


def _lin(k, n, g):
    m = torch.nn.Linear(k, n, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.randn(n, k, generator=g) * 0.02)
    return m


def _rope(position, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    ang = position * inv
    emb = torch.cat((ang, ang))
    return emb.cos().view(1, 1, 1, hd), emb.sin().view(1, 1, 1, hd)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Block(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.n1 = torch.nn.Parameter(torch.ones(DIM))
        self.n2 = torch.nn.Parameter(torch.ones(DIM))
        self.wq, self.wk, self.wv = _lin(DIM, HEADS * HD, g), _lin(DIM, KV_HEADS * HD, g), _lin(DIM, KV_HEADS * HD, g)
        self.wo = _lin(HEADS * HD, DIM, g)
        self.w1, self.w3, self.w2 = _lin(DIM, FFN, g), _lin(DIM, FFN, g), _lin(FFN, DIM, g)

    def forward(self, x, cos, sin, past_k, past_v):
        b = x.shape[0]
        y = F.rms_norm(x, (DIM,), self.n1, 1e-5)
        q = self.wq(y).view(b, 1, HEADS, HD).transpose(1, 2)
        k = self.wk(y).view(b, 1, KV_HEADS, HD).transpose(1, 2)
        v = self.wv(y).view(b, 1, KV_HEADS, HD).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        kk, vv = torch.cat((past_k, k), dim=2), torch.cat((past_v, v), dim=2)
        rep = HEADS // KV_HEADS
        kk = kk[:, :, None].expand(b, KV_HEADS, rep, kk.shape[2], HD).reshape(b, HEADS, -1, HD)
        vv = vv[:, :, None].expand(b, KV_HEADS, rep, vv.shape[2], HD).reshape(b, HEADS, -1, HD)
        o = F.scaled_dot_product_attention(q, kk, vv).transpose(1, 2).reshape(b, 1, HEADS * HD)
        x = x + self.wo(o)
        y = F.rms_norm(x, (DIM,), self.n2, 1e-5)
        return x + self.w2(F.silu(self.w1(y)) * self.w3(y)), k, v


class DecoderStep(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.tok = torch.nn.Parameter(torch.randn(VOCAB, DIM, generator=g) * 0.02)
        self.layers = torch.nn.ModuleList(Block(g) for _ in range(LAYERS))
        self.norm = torch.nn.Parameter(torch.ones(DIM))
        self.head = _lin(DIM, VOCAB, g)
        cos, sin = _rope(PAST, HD)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, token, past_k, past_v):
        x = F.embedding(token, self.tok)
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, self.cos, self.sin, past_k[i], past_v[i])
            ks.append(k)
            vs.append(v)
        return self.head(F.rms_norm(x, (DIM,), self.norm, 1e-5)), torch.stack(ks), torch.stack(vs)


def make():
    g = torch.Generator().manual_seed(0)
    model = DecoderStep(g).half().eval()
    shape = (LAYERS, BATCH, KV_HEADS, PAST, HD)
    return model, (torch.randint(0, VOCAB, (BATCH, 1), generator=g),
                   (torch.randn(shape, generator=g) * 0.5).half(), (torch.randn(shape, generator=g) * 0.5).half())
