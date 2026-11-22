"""op.add/membound/fp16-small-strided-r32-c1024

A point of the benchmark itself (op add, fp16), given to you as the evaluation runs it: the module
and inputs make() returns are the point's own (seed 0), judged at TOL, and DEMAND is the demand the
benchmark declares for it. Measure it with scripts/tnca_case.py like any case; copy it to vary a
dimension.

Generated from the benchmark at d8d6a4cdd5d6; regenerated, not edited.
"""
import torch
import bound  # this skill's scripts/: tnca_case.py puts it on the path

POINT = 'op.add/membound/fp16-small-strided-r32-c1024'
TOL = 0.02
DEMAND = bound.Demand(macs=0.0, fp_ops=32768, sfu_ops=0.0, read_bytes=131072, write_bytes=65536)

C = 1024
TORCH_DTYPE = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'e4m3': 'float8_e4m3fn', 'e5m2': 'float8_e5m2'}


class _Layout(torch.nn.Module):

    def __init__(self, layout, rows):
        super().__init__()
        self.layout = layout
        self.rows = rows

    def view(self, x):
        if self.layout == 'strided':
            return x[:, ::2]
        return x

    def last(self, x):
        if self.layout == 'broadcast':
            return x.expand(self.rows, x.shape[-1])
        return self.view(x)


class Binary(_Layout):

    def __init__(self, layout, rows, fn):
        super().__init__(layout, rows)
        self.fn = fn

    def forward(self, x, y):
        return self.fn(self.view(x), self.last(y))


def _input(g, layout, rows, dtype, last, divisor=False):
    if layout == 'broadcast' and last:
        shape, read = ((C,), C)
    elif layout == 'strided':
        shape, read = ((rows, 2 * C), rows * C)
    else:
        shape, read = ((rows, C), rows * C)
    return ((_divisor if divisor else _randn)(g, shape, dtype), read)


def _divisor(g, shape, dtype):
    r = torch.randn(shape, generator=g)
    return (torch.sign(r) * (0.5 + r.abs())).to(_td(dtype))


def _td(dtype):
    return getattr(torch, TORCH_DTYPE[dtype])


def _randn(g, shape, dtype):
    return torch.randn(shape, generator=g).to(_td(dtype))


def gen(seed):
    return torch.Generator().manual_seed(seed)


def _point_make(seed, layout='strided', rows=32, dtype='fp16', fn=torch.add, family='add'):
    g = gen(seed)
    x, _ = _input(g, layout, rows, dtype, last=False)
    y, _ = _input(g, layout, rows, dtype, last=True, divisor=family == 'div')
    return (Binary(layout, rows, fn), (x, y))


def make():
    return _point_make(0)
