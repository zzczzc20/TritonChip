"""op.attention_decode/tiny/fp16-b1-h2-l16-d16

A point of the benchmark itself (op attention_decode, fp16), given to you as the evaluation runs it:
the module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at dc80a07ca012; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.attention_decode/tiny/fp16-b1-h2-l16-d16'
TOL = 0.02
DEMAND = bound.Demand(macs=1024, fp_ops=96, sfu_ops=32, read_bytes=2112, write_bytes=64)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class DecodeAttention(torch.nn.Module):

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, b=1, h=2, l=16, d=16):
    return (DecodeAttention(), (randn(gen(seed), (b, h, 1, d), 'fp16'), randn(gen(seed + 1), (b, h, l, d), 'fp16'), randn(gen(seed + 2), (b, h, l, d), 'fp16')))


def make():
    return _point_make(0)
