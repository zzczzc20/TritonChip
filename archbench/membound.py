"""The R41 memory-bound tier (`membound`): the registered bandwidth suite.

evidence/r41/hbm_bw/BANDWIDTH_PLAN.md section 3 and doc/engineering_discipline.md
section 2.3 require the suite of a bandwidth target to be registered before any
design change is measured, and to cover the class of programs the target claims.
This module defines it; `python -m archbench.suite --freeze` writes it to
archbench/suites/r41_membound.json, and `--check` refuses any later edit.

Every point is a standard PyTorch callable on 2-D tensors of C = 1024 columns.
The rules below are generic: they are the same for every operator and dtype,
and they were fixed before any point of this tier was measured.

Operators (families):
    copy (dst.copy_(src) into an existing tensor), clone;
    add, sub, mul, div;
    relu, silu, gelu (the default, erf form), exp, sigmoid;
    cast between every ordered pair of fp32, fp16 and bf16;
    sum, mean, max (amax) over the last dimension;
    layernorm, rmsnorm, softmax over the last dimension.

Dtypes: fp32, fp16 and bf16 for every family (a cast's dtype is its source).

Sizes: the smallest full-size tensor of a point holds exactly
    small  64 KiB  (larger than the run-1 image's whole NUCA, 3 x 16 KiB)
    medium  4 MiB
    large  64 MiB  (BANDWIDTH_PLAN.md: at least 64 MiB per tensor)
so R = size / (1024 x the narrowest element size of the point's full-size
tensors). The output of a reduction (R values) is not a full-size tensor.

Layouts:
    contig     every full-size input is a contiguous (R, C) tensor;
    strided    every full-size input is read through a stride-2 column view,
               base[:, ::2], of a contiguous (R, 2C) base passed as the input
               (the view is taken inside the module, so no host transfer can
               make it contiguous);
    broadcast  the LAST full-size input is replaced by a (C,) row broadcast
               over the R rows (binary: the second operand; unary, cast,
               copy, clone, reductions and normalisations: their only data
               input, so the result is computed from an expanded row).

Demand (ArchBench semantics, registry.Demand): every input element the result
depends on read once (a strided view counts its R x C elements, not the
unused half of its base; a broadcast row counts C elements), every output
element written once. fp_ops and sfu_ops are per-element lower estimates;
undercounting only loosens the bound.

Inputs: randn from the point's seed; a divisor is sign(r) * (0.5 + |r|) so that
no quotient overflows fp16 (a property of the operator's domain, fixed here
before any device result). Tolerances: registry.TOLERANCE of the point's dtype;
a cast uses the larger tolerance of its two dtypes.

Nothing in the compiler, runtime or hardware may recognise these points
(discipline section 2.2.2); the tier only defines what is measured.
"""
import torch
import torch.nn.functional as F

from .ops import B, gen
from .registry import Demand, Point, TOLERANCE, TORCH_DTYPE, register

KiB, MiB = 1 << 10, 1 << 20
TIER = 'membound'
C = 1024
SIZES = (('small', 64 * KiB), ('medium', 4 * MiB), ('large', 64 * MiB))
LAYOUTS = ('contig', 'strided', 'broadcast')
DTYPES = ('fp32', 'fp16', 'bf16')


def _td(dtype):
    return getattr(torch, TORCH_DTYPE[dtype])


def _randn(g, shape, dtype):
    return torch.randn(shape, generator=g).to(_td(dtype))


def _divisor(g, shape, dtype):
    r = torch.randn(shape, generator=g)
    return (torch.sign(r) * (0.5 + r.abs())).to(_td(dtype))


class _Layout(torch.nn.Module):
    """Applies the layout rule to the data inputs, then the operator."""

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


class Unary(_Layout):
    def __init__(self, layout, rows, fn):
        super().__init__(layout, rows)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.last(x))


class Binary(_Layout):
    def __init__(self, layout, rows, fn):
        super().__init__(layout, rows)
        self.fn = fn

    def forward(self, x, y):
        # broadcast: only the second operand is a broadcast row
        return self.fn(self.view(x), self.last(y))


class Copy(_Layout):
    """dst.copy_(src): the output is an existing tensor, so dst is not read."""

    def forward(self, dst, src):
        return dst.copy_(self.last(src))


class Norm(_Layout):
    def __init__(self, layout, rows, kind, dtype, g):
        super().__init__(layout, rows)
        self.kind = kind
        self.weight = torch.nn.Parameter(1 + 0.1 * _randn(g, (C,), dtype).float().to(_td(dtype)))
        if kind == 'layernorm':
            self.bias = torch.nn.Parameter(0.1 * _randn(g, (C,), dtype).float().to(_td(dtype)))

    def forward(self, x):
        x = self.last(x)
        if self.kind == 'layernorm':
            return F.layer_norm(x, (C,), self.weight, self.bias, 1e-5)
        return F.rms_norm(x, (C,), self.weight, 1e-5)


def _relu(x):
    return torch.relu(x)


def _silu(x):
    return F.silu(x)


def _gelu(x):
    return F.gelu(x)


def _exp(x):
    return torch.exp(x)


def _sigmoid(x):
    return torch.sigmoid(x)


def _clone(x):
    return x.clone()


def _sum(x):
    return x.sum(dim=-1)


def _mean(x):
    return x.mean(dim=-1)


def _max(x):
    return x.amax(dim=-1)


def _softmax(x):
    return torch.softmax(x, dim=-1)


def _cast_to(dtype):
    target = _td(dtype)

    def cast(x):
        return x.to(target)
    return cast


# family -> (kind, callable, fp_ops per element, sfu_ops per element)
UNARY = dict(clone=(_clone, 0, 0), relu=(_relu, 1, 0), silu=(_silu, 2, 1), gelu=(_gelu, 4, 1),
             exp=(_exp, 0, 1), sigmoid=(_sigmoid, 1, 1))
BINARY = dict(add=(torch.add, 1), sub=(torch.sub, 1), mul=(torch.mul, 1), div=(torch.div, 1))
REDUCE = dict(sum=(_sum, 1), mean=(_mean, 1), max=(_max, 1))


def _rows(size_bytes, *dtypes):
    return size_bytes // (C * min(B[d] for d in dtypes))


def _input(g, layout, rows, dtype, last, divisor=False):
    """One data input of the point: (tensor, logical elements read)."""
    if layout == 'broadcast' and last:
        shape, read = (C,), C
    elif layout == 'strided':
        shape, read = (rows, 2 * C), rows * C
    else:
        shape, read = (rows, C), rows * C
    return (_divisor if divisor else _randn)(g, shape, dtype), read


def _point(family, dtype, size, layout, rows, make, demand, *, tol=None, to=None):
    head = f'{dtype}-to-{to}' if to else dtype
    pid = f'op.{family}/{TIER}/{head}-{size}-{layout}-r{rows}-c{C}'
    params = dict(size=size, layout=layout, r=rows, c=C)
    if to:
        params['to'] = to
    return register(Point(pid, 'op', family, TIER, dtype, params, make, demand,
                          track=('llm', 'general'), tol=tol, note='R41 memory-bound suite'))


def _register():
    for size, nbytes in SIZES:
        for layout in LAYOUTS:
            for dtype in DTYPES:
                b = B[dtype]
                rows = _rows(nbytes, dtype)
                n = rows * C
                # copy: dst (R, C) written, src read
                def mk_copy(seed, layout=layout, rows=rows, dtype=dtype):
                    g = gen(seed)
                    dst = torch.zeros((rows, C), dtype=_td(dtype))
                    src, _ = _input(g, layout, rows, dtype, last=True)
                    return Copy(layout, rows), (dst, src)
                read = C if layout == 'broadcast' else n
                _point('copy', dtype, size, layout, rows, mk_copy,
                       Demand(read_bytes=read * b, write_bytes=n * b))
                for family, (fn, fp, sfu) in UNARY.items():
                    def mk_unary(seed, layout=layout, rows=rows, dtype=dtype, fn=fn):
                        g = gen(seed)
                        x, _ = _input(g, layout, rows, dtype, last=True)
                        return Unary(layout, rows, fn), (x,)
                    _point(family, dtype, size, layout, rows, mk_unary,
                           Demand(fp_ops=fp * n, sfu_ops=sfu * n, read_bytes=read * b, write_bytes=n * b))
                for family, (fn, fp) in BINARY.items():
                    def mk_binary(seed, layout=layout, rows=rows, dtype=dtype, fn=fn, family=family):
                        g = gen(seed)
                        x, _ = _input(g, layout, rows, dtype, last=False)
                        y, _ = _input(g, layout, rows, dtype, last=True, divisor=family == 'div')
                        return Binary(layout, rows, fn), (x, y)
                    _point(family, dtype, size, layout, rows, mk_binary,
                           Demand(fp_ops=fp * n, read_bytes=(n + read) * b, write_bytes=n * b))
                for family, (fn, fp) in REDUCE.items():
                    def mk_reduce(seed, layout=layout, rows=rows, dtype=dtype, fn=fn):
                        g = gen(seed)
                        x, _ = _input(g, layout, rows, dtype, last=True)
                        return Unary(layout, rows, fn), (x,)
                    _point(family, dtype, size, layout, rows, mk_reduce,
                           Demand(fp_ops=fp * n, read_bytes=read * b, write_bytes=rows * b))
                for family, fp, sfu, params in (('layernorm', 4, 0, 2), ('rmsnorm', 3, 0, 1),
                                                ('softmax', 3, 1, 0)):
                    if family == 'softmax':
                        def mk_norm(seed, layout=layout, rows=rows, dtype=dtype):
                            g = gen(seed)
                            x, _ = _input(g, layout, rows, dtype, last=True)
                            return Unary(layout, rows, _softmax), (x,)
                    else:
                        def mk_norm(seed, layout=layout, rows=rows, dtype=dtype, family=family):
                            g = gen(seed)
                            x, _ = _input(g, layout, rows, dtype, last=True)
                            return Norm(layout, rows, family, dtype, g), (x,)
                    # one rsqrt per row for the normalisations, one exp per element for softmax
                    _point(family, dtype, size, layout, rows, mk_norm,
                           Demand(fp_ops=fp * n, sfu_ops=sfu * n if sfu else rows,
                                  read_bytes=(read + params * C) * b, write_bytes=n * b))
            for src in DTYPES:
                for dst in DTYPES:
                    if src == dst:
                        continue
                    rows = _rows(nbytes, src, dst)
                    n = rows * C
                    read = C if layout == 'broadcast' else n

                    def mk_cast(seed, layout=layout, rows=rows, src=src, dst=dst):
                        g = gen(seed)
                        x, _ = _input(g, layout, rows, src, last=True)
                        return Unary(layout, rows, _cast_to(dst)), (x,)
                    _point('cast', src, size, layout, rows, mk_cast,
                           Demand(read_bytes=read * B[src], write_bytes=n * B[dst]),
                           tol=max(TOLERANCE[src], TOLERANCE[dst]), to=dst)


_register()
