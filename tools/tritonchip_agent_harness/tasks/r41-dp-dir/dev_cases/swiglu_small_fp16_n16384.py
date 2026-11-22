"""op.swiglu/small/fp16-n16384

A point of the benchmark itself (op swiglu, fp16), given to you as the evaluation runs it: the
module and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the
demand the benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to
vary a dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import torch.nn.functional as F
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.swiglu/small/fp16-n16384'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=49152, sfu_ops=16384, read_bytes=65536, write_bytes=32768)

dtype = 'fp16'
n_inputs = 2
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class SwiGLU(torch.nn.Module):

    def forward(self, gate, up):
        return F.silu(gate) * up


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))

module = SwiGLU


def _point_make(seed, n=16384):
    g = gen(seed)
    return (module(), tuple((randn(g, (n,), dtype) for _ in range(n_inputs))))


def make():
    return _point_make(0)
