#!/usr/bin/env python3
"""
The two Phase 3.5 workloads that R3 v2 did not already have.

doc/phase3_5_tasks_and_acceptance.md section 6.1 freezes six sources.
L01, L04, L05 and L06 are C02, C06, C03 and C05 from the R3 v2
inventory, reused deliberately: the same programs on the same hardware,
so a cycle difference is the compiler's and nothing else's. The two here
are new because the stage needs them and nothing else in the tree does.

Section 6.1 also sets the rule these are written under: "不得手写差的 B0
汇编或添加死 copy 制造胜利；优化前先证明源表达的是实际计算需求". Neither
of these computes anything it does not need. L02's two outputs are both
checked, so neither can be dropped; L03's view composition is a real
reindex of the data it then consumes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'r3v2'))


# ---------------------------------------------------------------------------
# L02: two outputs sharing one intermediate
# ---------------------------------------------------------------------------

def triton_shared_intermediate():
    """t = x*alpha; y = t + b; z = t - b, both outputs stored.

    The formula section 6.1 names. The sharing is in the source: `t` is
    computed once and read twice, which is how anybody would write it.
    What a compiler can still find here is the rest -- one offset vector,
    one mask, and two stores that each recompute their address.

    Both y and z are checked by the oracle, so removing either is a wrong
    answer rather than a fast one.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def l02_shared(x_ptr, b_ptr, y_ptr, z_ptr, n, alpha,
                   BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        b = tl.load(b_ptr + offs, mask=m, other=0.0)
        t = x * alpha
        tl.store(y_ptr + offs, t + b, mask=m)
        tl.store(z_ptr + offs, t - b, mask=m)

    return l02_shared


def l02_oracle(xs, bs, n, alpha):
    """What both outputs must hold, in double, rounded once at the end."""
    import struct

    def f32(v):
        return struct.unpack('<f', struct.pack('<f', v))[0]
    t = [f32(f32(xs[i]) * f32(alpha)) for i in range(n)]
    return ([f32(t[i] + f32(bs[i])) for i in range(n)],
            [f32(t[i] - f32(bs[i])) for i in range(n)])


# ---------------------------------------------------------------------------
# L03: a legal view composition, then elementwise
# ---------------------------------------------------------------------------
#
# Section 6.1 puts L03 on the tnca-ir input and section 3.2 says what a
# layout conversion is allowed to be. This one is category 1: a pure
# compile-time reindex. The 4x4 tile is read as rows, consumed as
# columns, and on a machine with sixteen lanes and no cross-lane
# permutation instruction the honest lowering of a transpose is either a
# rename -- when the consumer does not care which lane holds which
# element -- or a refusal.
#
# So this is written as the non-degenerate control section 6.1 asks for:
# "若纯 rename 已经零成本，仍保留为不退化对照，不能强迫基线搬运". The
# baseline is not made to move data it never needed to move.

def tnca_view_elementwise(rows=4, cols=4):
    """Read a tile, view it transposed, view it back, consume it.

    Section 6.1 puts L03 on the tnca-ir input and calls it "a legal
    tensor view/layout composition, then elementwise consumption". This
    is that composition: the tile is read as rows and columns, then read
    back the way it came.

    The two views cancel. Whether anything NOTICES they cancel is the
    whole question, and it is the question section 6.3's third condition
    asks -- so this is the predetermined ablation subject. With the
    layout algebra on, upstream composes them, the composition is the
    identity over all sixteen positions, and no value moves. With it
    off, each view is materialised: an index vector built from the lane
    index and a `v.permute`, twice.

    Neither path removes any computation. The elementwise work happens
    either way and both produce the same numbers, which is what makes
    the cycle difference attributable to the layout decision and
    nothing else.
    """
    n = rows * cols
    # reg -> (row, col): bit k of the lane index selects basis k.
    fwd = [('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])]
    # (row, col) -> reg, reading the tile back the way it came.
    back = [('row', [[1], [2]]), ('col', [[4], [8]])]

    def build(b, target):
        base = b.special('arg_base')
        p_in = b.sload(base, 0, 'd')
        p_out = b.sload(base, 8, 'd')

        idx = b.arange()
        keep = b.vcmp('lt', idx, b.splat(b.const(n)))
        v = b.load(p_in, mask=keep, other=0, dtype='f32')

        t1 = b.view(v, fwd, ['row', 'col'], [rows, cols],
                    what='row-major as column-major')
        t2 = b.view(t1, back, ['reg'], [n], what='and back again')

        # The elementwise consumption: doubling, which needs no constant
        # and is the same in every lane, so nothing about the result
        # depends on which lane holds which element except the layout.
        doubled = b.vbin('add', t2, t2, dtype='f32')
        b.store(p_out, doubled, mask=keep)
        b.exit()
    return build


def l03_oracle(xs, rows=4, cols=4):
    """Two cancelling views leave the data where it was, then doubled."""
    import struct

    def f32(v):
        return struct.unpack('<f', struct.pack('<f', v))[0]
    return [f32(f32(xs[i]) + f32(xs[i])) for i in range(rows * cols)]
