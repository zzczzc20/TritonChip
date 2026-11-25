"""op.add/hbm/fp16-n67108864

A point of the benchmark itself (op add, fp16), given to you as the evaluation runs it: the module
and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the demand the
benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to vary a
dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.add/hbm/fp16-n67108864'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=67108864, sfu_ops=0, read_bytes=268435456, write_bytes=134217728)

TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class Add(torch.nn.Module):

    def forward(self, x, y):
        return x + y


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, shape, dtype, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(getattr(torch, TORCH_DTYPE[dtype]))


def _point_make(seed, module=Add, n=67108864, n_in=2):
    g = gen(seed)
    return (module(), tuple((randn(g, (n,), 'fp16') for _ in range(n_in))))


def make():
    return _point_make(0)
