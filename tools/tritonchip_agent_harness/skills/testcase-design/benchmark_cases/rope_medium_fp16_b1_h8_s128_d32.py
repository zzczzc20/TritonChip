"""op.rope/medium/fp16-b1-h8-s128-d32

A point of the benchmark itself (op rope, fp16), given to you as the evaluation runs it: the module
and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the demand the
benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to vary a
dimension.

Generated from the benchmark at a3e73d08f865; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.rope/medium/fp16-b1-h8-s128-d32'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=98304, sfu_ops=0.0, read_bytes=81920, write_bytes=65536)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Rope(torch.nn.Module):
    def forward(self, x, cos, sin):
        return x * cos + rotate_half(x) * sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def rope_tables(s, d, dtype, offset=0, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    t = torch.arange(offset, offset + s, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    dt = getattr(torch, TORCH_DTYPE[dtype])
    return emb.cos().to(dt), emb.sin().to(dt)


def _point_make(seed, b=1, h=8, s=128, d=32):
    cos, sin = rope_tables(s, d, 'fp16')
    return (Rope(), (randn(gen(seed), (b, h, s, d), 'fp16'), cos, sin))


def make():
    return _point_make(0)
