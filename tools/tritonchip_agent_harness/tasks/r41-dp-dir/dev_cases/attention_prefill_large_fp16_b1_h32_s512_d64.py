"""op.attention_prefill/large/fp16-b1-h32-s512-d64

A point of the benchmark itself (op attention_prefill, fp16), given to you as the evaluation runs
it: the module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is
the demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy
it to vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.attention_prefill/large/fp16-b1-h32-s512-d64'
TOL = 0.02
DEMAND = bound.Demand(macs=537919488, fp_ops=12607488, sfu_ops=4202496, read_bytes=6291456, write_bytes=2097152)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class CausalAttention(torch.nn.Module):

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, b=1, h=32, s=512, d=64):
    return (CausalAttention(), tuple((randn(gen(seed + i), (b, h, s, d), 'fp16') for i in range(3))))


def make():
    return _point_make(0)
