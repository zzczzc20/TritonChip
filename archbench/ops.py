"""Operator points: the torch ops an LLM (and a little beyond) spends its time in.

Each family is a torch-level callable plus a demand formula. The demand is
the SEMANTIC minimum (roadmap section 4.4): MACs by the conventional dense
count, causal attention counting only unmasked products, bytes = every input
the result depends on read once plus every output written once. Undercounting
is safe (it only loosens the bound); overcounting would let a platform beat
its own bound, which the scorer refuses rather than clips.
"""
import torch
import torch.nn.functional as F

from .registry import Demand, Point, TORCH_DTYPE, register

B = {'fp32': 4, 'fp16': 2, 'bf16': 2}
I64 = 8


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _fmt(**kw):
    return '-'.join(f'{k}{v}' for k, v in kw.items())


def op(family, tier, dtype, params, make, demand, *, track=('llm',), tol=None, note=''):
    pid = f'op.{family}/{tier}/{dtype}-{_fmt(**params)}'
    return register(Point(pid, 'op', family, tier, dtype, dict(params), make, demand,
                          track=tuple(track), tol=tol, note=note))


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

class Add(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class SiLU(torch.nn.Module):
    def forward(self, x):
        return F.silu(x)


class GeLUTanh(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x, approximate='tanh')


class SwiGLU(torch.nn.Module):
    def forward(self, gate, up):
        return F.silu(gate) * up


def _elementwise(family, module, n_inputs, fp, sfu, sizes, dtype='fp16', track=('llm',)):
    for tier, n in sizes.items():
        def make(seed, n=n):
            g = gen(seed)
            return module(), tuple(randn(g, (n,), dtype) for _ in range(n_inputs))
        b = B[dtype]
        op(family, tier, dtype, dict(n=n), make,
           Demand(fp_ops=fp * n, sfu_ops=sfu * n, read_bytes=n_inputs * n * b, write_bytes=n * b),
           track=track)


_elementwise('add', Add, 2, 1, 0, dict(tiny=1000, small=16384, medium=49152, large=1 << 24))
_elementwise('silu', SiLU, 1, 2, 1, dict(tiny=256, small=16384, medium=65536, large=1 << 24))
_elementwise('gelu_tanh', GeLUTanh, 1, 4, 1, dict(tiny=256, small=16384, medium=65536, large=1 << 24))
_elementwise('swiglu', SwiGLU, 2, 3, 1, dict(tiny=256, small=16384, medium=49152, large=1 << 24))
_elementwise('add', Add, 2, 1, 0, dict(tiny=256), dtype='fp32', track=('general',))


# ---------------------------------------------------------------------------
# Row reductions: softmax, norms, argmax
# ---------------------------------------------------------------------------

class Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(1 + randn(g, (d,), dtype, 0.1))
        self.eps = 1e-5

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class LayerNorm(torch.nn.Module):
    def __init__(self, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(1 + randn(g, (d,), dtype, 0.1))
        self.bias = torch.nn.Parameter(randn(g, (d,), dtype, 0.1))

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, 1e-5)


class Argmax(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=-1)


ROWS = dict(tiny=(4, 64), small=(64, 512), medium=(80, 512), large=(4096, 4096))
for tier, (r, c) in ROWS.items():
    n = r * c
    op('softmax', tier, 'fp16', dict(r=r, c=c),
       lambda seed, r=r, c=c: (Softmax(), (randn(gen(seed), (r, c), 'fp16', 2.0),)),
       Demand(fp_ops=3 * n, sfu_ops=n, read_bytes=n * 2, write_bytes=n * 2))
op('softmax', 'tiny', 'fp32', dict(r=3, c=37),
   lambda seed: (Softmax(), (randn(gen(seed), (3, 37), 'fp32', 2.0),)),
   Demand(fp_ops=3 * 111, sfu_ops=111, read_bytes=111 * 4, write_bytes=111 * 4),
   track=('general',), note='a row length that is not a multiple of the vector length')

NORM_ROWS = dict(tiny=(4, 64), small=(64, 256), medium=(128, 512), large=(512, 2048))
for tier, (r, d) in NORM_ROWS.items():
    n = r * d

    def mk_rms(seed, r=r, d=d):
        g = gen(seed)
        return RMSNorm(d, 'fp16', g), (randn(g, (r, d), 'fp16'),)

    def mk_ln(seed, r=r, d=d):
        g = gen(seed)
        return LayerNorm(d, 'fp16', g), (randn(g, (r, d), 'fp16'),)
    op('rmsnorm', tier, 'fp16', dict(r=r, d=d), mk_rms,
       Demand(fp_ops=3 * n, sfu_ops=r, read_bytes=(n + d) * 2, write_bytes=n * 2))
    op('layernorm', tier, 'fp16', dict(r=r, d=d), mk_ln,
       Demand(fp_ops=4 * n, sfu_ops=r, read_bytes=(n + 2 * d) * 2, write_bytes=n * 2),
       track=('llm', 'general'))

ARGMAX = dict(tiny=(2, 256), small=(8, 2048), medium=(32, 4096), large=(64, 128256))
for tier, (r, v) in ARGMAX.items():
    op('argmax', tier, 'fp16', dict(r=r, v=v),
       lambda seed, r=r, v=v: (Argmax(), (randn(gen(seed), (r, v), 'fp16', 3.0),)),
       Demand(fp_ops=r * v, read_bytes=r * v * 2, write_bytes=r * I64),
       tol=0.0, note='greedy sampling over the vocabulary; exact')


# ---------------------------------------------------------------------------
# GEMM family
# ---------------------------------------------------------------------------

class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class BMM(torch.nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)


def _linear(k, n, dtype, g, bias):
    lin = torch.nn.Linear(k, n, bias=bias)
    with torch.no_grad():
        lin.weight.copy_(randn(g, (n, k), 'fp32', k ** -0.5))
        if bias:
            lin.bias.copy_(randn(g, (n,), 'fp32', 0.1))
    return lin.to(getattr(torch, TORCH_DTYPE[dtype]))


MM = dict(tiny=(16, 32, 16), small=(64, 128, 128), medium=(128, 192, 128), large=(2048, 2048, 2048))
for dtype in ('fp16', 'fp32'):
    for tier, (m, k, n) in MM.items():
        b = B[dtype]
        op('matmul', tier, dtype, dict(m=m, k=k, n=n),
           lambda seed, m=m, k=k, n=n, dtype=dtype: (
               MatMul(), (randn(gen(seed), (m, k), dtype), randn(gen(seed + 1), (k, n), dtype))),
           Demand(macs=m * k * n, read_bytes=(m * k + k * n) * b, write_bytes=m * n * b),
           track=('llm', 'general') if dtype == 'fp16' else ('general',))

LINEAR = dict(tiny=(16, 64, 64), small=(64, 256, 256), medium=(128, 320, 256), large=(512, 2048, 8192))
for tier, (m, k, n) in LINEAR.items():
    def mk(seed, m=m, k=k, n=n):
        g = gen(seed)
        return _linear(k, n, 'fp16', g, True), (randn(g, (m, k), 'fp16'),)
    op('linear', tier, 'fp16', dict(m=m, k=k, n=n), mk,
       Demand(macs=m * k * n, fp_ops=m * n, read_bytes=(m * k + k * n + n) * 2, write_bytes=m * n * 2))

GEMV = dict(tiny=(64, 64), small=(256, 512), medium=(320, 448), large=(2048, 8192))
for tier, (k, n) in GEMV.items():
    def mk(seed, k=k, n=n):
        g = gen(seed)
        return _linear(k, n, 'fp16', g, False), (randn(g, (1, k), 'fp16'),)
    op('gemv', tier, 'fp16', dict(k=k, n=n), mk,
       Demand(macs=k * n, read_bytes=(k + k * n) * 2, write_bytes=n * 2),
       note='decode-time projection: one token, weight-bandwidth bound')

BMMS = dict(tiny=(2, 16, 16, 16), small=(8, 64, 32, 64), medium=(16, 64, 32, 64), large=(32, 512, 64, 512))
for tier, (bt, m, k, n) in BMMS.items():
    op('bmm', tier, 'fp16', dict(b=bt, m=m, k=k, n=n),
       lambda seed, bt=bt, m=m, k=k, n=n: (
           BMM(), (randn(gen(seed), (bt, m, k), 'fp16'), randn(gen(seed + 1), (bt, k, n), 'fp16'))),
       Demand(macs=bt * m * k * n, read_bytes=bt * (m * k + k * n) * 2, write_bytes=bt * m * n * 2))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class DecodeAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


PREFILL = dict(tiny=(1, 2, 16, 16), small=(1, 4, 64, 32), medium=(1, 4, 96, 32), large=(1, 32, 512, 64))
for tier, (b, h, s, d) in PREFILL.items():
    tri = s * (s + 1) // 2
    op('attention_prefill', tier, 'fp16', dict(b=b, h=h, s=s, d=d),
       lambda seed, b=b, h=h, s=s, d=d: (CausalAttention(), tuple(
           randn(gen(seed + i), (b, h, s, d), 'fp16') for i in range(3))),
       Demand(macs=2 * b * h * d * tri, fp_ops=3 * b * h * tri, sfu_ops=b * h * tri,
              read_bytes=3 * b * h * s * d * 2, write_bytes=b * h * s * d * 2),
       note='causal: only the s(s+1)/2 unmasked scores are work')

DECODE = dict(tiny=(1, 2, 16, 16), small=(1, 4, 256, 32), medium=(1, 8, 256, 32), large=(1, 32, 2048, 64))
for tier, (b, h, l, d) in DECODE.items():
    op('attention_decode', tier, 'fp16', dict(b=b, h=h, l=l, d=d),
       lambda seed, b=b, h=h, l=l, d=d: (DecodeAttention(), (
           randn(gen(seed), (b, h, 1, d), 'fp16'),
           randn(gen(seed + 1), (b, h, l, d), 'fp16'),
           randn(gen(seed + 2), (b, h, l, d), 'fp16'))),
       Demand(macs=2 * b * h * l * d, fp_ops=3 * b * h * l, sfu_ops=b * h * l,
              read_bytes=(b * h * d + 2 * b * h * l * d) * 2, write_bytes=b * h * d * 2),
       note='one query token against an l-entry KV cache')


# ---------------------------------------------------------------------------
# Data movement and the rest of a decoder
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(torch.nn.Module):
    def forward(self, x, cos, sin):
        return x * cos + rotate_half(x) * sin


def rope_tables(s, d, dtype, offset=0, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    t = torch.arange(offset, offset + s, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    dt = getattr(torch, TORCH_DTYPE[dtype])
    return emb.cos().to(dt), emb.sin().to(dt)


ROPE = dict(tiny=(1, 2, 16, 16), small=(1, 4, 64, 32), medium=(1, 8, 128, 32), large=(1, 32, 512, 64))
for tier, (b, h, s, d) in ROPE.items():
    n = b * h * s * d

    def mk(seed, b=b, h=h, s=s, d=d):
        cos, sin = rope_tables(s, d, 'fp16')
        return Rope(), (randn(gen(seed), (b, h, s, d), 'fp16'), cos, sin)
    op('rope', tier, 'fp16', dict(b=b, h=h, s=s, d=d), mk,
       Demand(fp_ops=3 * n, read_bytes=(n + 2 * s * d) * 2, write_bytes=n * 2))


class Embedding(torch.nn.Module):
    def __init__(self, v, d, dtype, g):
        super().__init__()
        self.weight = torch.nn.Parameter(randn(g, (v, d), dtype))

    def forward(self, ids):
        return F.embedding(ids, self.weight)


EMB = dict(tiny=(256, 64, 16), small=(1024, 128, 64), medium=(1024, 128, 256), large=(128256, 2048, 512))
for tier, (v, d, t) in EMB.items():
    def mk(seed, v=v, d=d, t=t):
        g = gen(seed)
        # Distinct ids, so the rows the result needs are exactly t rows.
        ids = torch.randperm(v, generator=g)[:t]
        return Embedding(v, d, 'fp16', g), (ids,)
    op('embedding', tier, 'fp16', dict(v=v, d=d, t=t), mk,
       Demand(read_bytes=t * I64 + t * d * 2, write_bytes=t * d * 2), tol=0.0,
       note='a gather: only the t rows looked up are required reads')


class Transpose(torch.nn.Module):
    def forward(self, x):
        return x.t().contiguous()


TR = dict(tiny=(32, 32), small=(128, 256), medium=(192, 256), large=(4096, 4096))
for tier, (r, c) in TR.items():
    op('transpose', tier, 'fp32', dict(r=r, c=c),
       lambda seed, r=r, c=c: (Transpose(), (randn(gen(seed), (r, c), 'fp32'),)),
       Demand(read_bytes=r * c * 4, write_bytes=r * c * 4), tol=0.0, track=('general',))


class Cumsum(torch.nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=-1)


for tier, (r, c) in dict(tiny=(4, 64), small=(64, 256), medium=(128, 256)).items():
    op('cumsum', tier, 'fp32', dict(r=r, c=c),
       lambda seed, r=r, c=c: (Cumsum(), (randn(gen(seed), (r, c), 'fp32'),)),
       Demand(fp_ops=r * c, read_bytes=r * c * 4, write_bytes=r * c * 4), tol=1e-3,
       track=('general',))


class Conv(torch.nn.Module):
    def __init__(self, cin, cout, g):
        super().__init__()
        self.conv = torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(randn(g, (cout, cin, 3, 3), 'fp32', (9 * cin) ** -0.5))

    def forward(self, x):
        return self.conv(x)


CONV = dict(tiny=(1, 4, 8, 8, 8), small=(1, 16, 16, 16, 16), medium=(1, 24, 16, 16, 24), large=(8, 64, 56, 56, 64))
for tier, (nb, cin, h, w, cout) in CONV.items():
    def mk(seed, nb=nb, cin=cin, h=h, w=w, cout=cout):
        g = gen(seed)
        return Conv(cin, cout, g), (randn(g, (nb, cin, h, w), 'fp32'),)
    op('conv2d', tier, 'fp32', dict(n=nb, cin=cin, h=h, w=w, cout=cout), mk,
       Demand(macs=nb * cout * cin * (3 * h - 2) * (3 * w - 2),
              read_bytes=(nb * cin * h * w + cout * cin * 9) * 4, write_bytes=nb * cout * h * w * 4),
       track=('general',), note='3x3, padding 1: only taps inside the image are MACs')
