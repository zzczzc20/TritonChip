"""model.llama_decode/medium/fp16-b1-p192

A point of the benchmark itself (model llama_decode, fp16), given to you as the evaluation runs it:
the module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import _thread
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'model.llama_decode/medium/fp16-b1-p192'
TOL = 0.05
DEMAND = bound.Demand(macs=1057536, fp_ops=0.0, sfu_ops=2316, read_bytes=1919240, write_bytes=1792)
MODEL = True

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


@dataclass(frozen=True)
class LlamaCfg:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn: int
    vocab: int
    eps: float = 1e-05

    @property
    def hd(self):
        return self.dim // self.n_heads


class Llama(torch.nn.Module):

    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.tok = torch.nn.Parameter(randn(g, (c.vocab, c.dim), dtype, 0.02))
        self.layers = torch.nn.ModuleList((LlamaBlock(c, g, dtype) for _ in range(c.n_layers)))
        self.norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.out = _lin(c.dim, c.vocab, g, dtype)
        self.dtype = dtype

    def _head(self, x):
        return self.out(F.rms_norm(x, (self.c.dim,), self.norm, self.c.eps))


class LlamaBlock(torch.nn.Module):

    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.attn_norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.attn = LlamaAttention(c, g, dtype)
        self.mlp_norm = torch.nn.Parameter(torch.ones(c.dim, dtype=_dt(dtype)))
        self.w1 = _lin(c.dim, c.ffn, g, dtype)
        self.w3 = _lin(c.dim, c.ffn, g, dtype)
        self.w2 = _lin(c.ffn, c.dim, g, dtype)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        h, nk, nv = self.attn(F.rms_norm(x, (self.c.dim,), self.attn_norm, self.c.eps), cos, sin, past_k, past_v)
        x = x + h
        y = F.rms_norm(x, (self.c.dim,), self.mlp_norm, self.c.eps)
        x = x + self.w2(F.silu(self.w1(y)) * self.w3(y))
        return (x, nk, nv)


class LlamaAttention(torch.nn.Module):

    def __init__(self, c, g, dtype):
        super().__init__()
        self.c = c
        self.wq = _lin(c.dim, c.n_heads * c.hd, g, dtype)
        self.wk = _lin(c.dim, c.n_kv_heads * c.hd, g, dtype)
        self.wv = _lin(c.dim, c.n_kv_heads * c.hd, g, dtype)
        self.wo = _lin(c.n_heads * c.hd, c.dim, g, dtype)

    def forward(self, x, cos, sin, past_k=None, past_v=None):
        c = self.c
        b, s, _ = x.shape
        q = self.wq(x).view(b, s, c.n_heads, c.hd).transpose(1, 2)
        k = self.wk(x).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
        v = self.wv(x).view(b, s, c.n_kv_heads, c.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        new_k, new_v = (k, v)
        if past_k is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        rep = c.n_heads // c.n_kv_heads
        if rep > 1:
            k = k[:, :, None].expand(b, c.n_kv_heads, rep, k.shape[2], c.hd).reshape(b, c.n_heads, -1, c.hd)
            v = v[:, :, None].expand(b, c.n_kv_heads, rep, v.shape[2], c.hd).reshape(b, c.n_heads, -1, c.hd)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=past_k is None)
        o = o.transpose(1, 2).reshape(b, s, c.n_heads * c.hd)
        return (self.wo(o), new_k, new_v)


def _lin(k, n, g, dtype, bias=False, std=0.02):
    m = torch.nn.Linear(k, n, bias=bias)
    with torch.no_grad():
        m.weight.copy_(randn(g, (n, k), 'fp32', std))
        if bias:
            m.bias.copy_(randn(g, (n,), 'fp32', std))
    return m.to(_dt(dtype))


def _dt(dtype):
    return getattr(torch, TORCH_DTYPE[dtype])


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LlamaDecode(Llama):

    def __init__(self, c, g, dtype, p):
        super().__init__(c, g, dtype)
        cos, sin = rope_tables(1, c.hd, dtype, offset=p)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, token, past_k, past_v):
        x = F.embedding(token, self.tok)
        nks, nvs = ([], [])
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, self.cos, self.sin, past_k[i], past_v[i])
            nks.append(nk)
            nvs.append(nv)
        return (self._head(x), torch.stack(nks), torch.stack(nvs))


def rope_tables(s, d, dtype, offset=0, theta=10000.0):
    inv = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
    t = torch.arange(offset, offset + s, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    dt = getattr(torch, TORCH_DTYPE[dtype])
    return (emb.cos().to(dt), emb.sin().to(dt))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed, c=LlamaCfg(dim=192, n_layers=2, n_heads=6, n_kv_heads=2, ffn=512, vocab=640, eps=1e-05), bsz=1, p=192):
    g = gen(seed)
    m = LlamaDecode(c, g, 'fp16', p).eval()
    shape = (c.n_layers, bsz, c.n_kv_heads, p, c.hd)
    return (m, (torch.randint(0, c.vocab, (bsz, 1), generator=g), randn(g, shape, 'fp16'), randn(g, shape, 'fp16')))


def make():
    return _point_make(0)
