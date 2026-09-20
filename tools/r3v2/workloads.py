"""
The six frozen workloads -- doc/phase3_tasks_and_acceptance.md section 4.1.

One definition each, compiled through the product driver and run through
the product runtime, so C01 on the reference and C01 on the candidate
differ in exactly one thing: which register file the device was elaborated
with.

Every expected value is computed here in Python from the inputs. Section
4.1 is explicit that reference and candidate agreeing is not enough --
both have to agree with an independent oracle, because two runs of the
same wrong thing agree perfectly.
"""
import os
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'compiler', 'python'))

SENTINEL = 0x7FC00000          # a quiet NaN: an untouched slot shows up
CANARY = b'\xA5' * 64
SEED = 42


def f32(values):
    return b''.join(struct.pack('<f', v) for v in values)


def unf32(blob):
    return list(struct.unpack(f'<{len(blob) // 4}f', blob))


def f16_bits(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def inputs(n, seed=SEED, lo=-8, hi=8):
    """Small exact integers in binary32 and binary16.

    The point of these cases is the register file, not floating-point
    association: a mismatch should mean the wrong value arrived, not that
    two correct roundings differ.
    """
    import random
    rng = random.Random(seed)
    return [float(rng.randint(lo, hi)) for _ in range(n)]


# ---------------------------------------------------------------------------
# C01 / C02: elementwise, through the real Triton frontend
# ---------------------------------------------------------------------------

def triton_vecadd():
    import triton
    import triton.language as tl

    @triton.jit
    def c01_vecadd(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        y = tl.load(y_ptr + offs, mask=m, other=0.0)
        tl.store(o_ptr + offs, x + y, mask=m)

    return c01_vecadd


def triton_masked_axpy():
    import triton
    import triton.language as tl

    @triton.jit
    def c02_masked_axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        y = tl.load(y_ptr + offs, mask=m, other=0.0)
        tl.store(o_ptr + offs, x * alpha + y, mask=m)

    return c02_masked_axpy


# ---------------------------------------------------------------------------
# C06: a masked reduction, in TNCA IR
# ---------------------------------------------------------------------------

def tnca_reduction(valid):
    """Sum the first `valid` lanes of one vector register.

    Written against the builder rather than through Triton because the
    contract puts C06 on the tnca-ir input, and because the mask and the
    reduction order are then this file's decision rather than a frontend's.
    """
    def build(b, target):
        base = b.special('arg_base')
        p_in = b.sload(base, 0, 'd')
        p_out = b.sload(base, 8, 'd')

        idx = b.arange()
        keep = b.vcmp('lt', idx, b.splat(b.const(valid)))
        v = b.load(p_in, mask=keep, other=0, dtype='f32')
        total = b.reduce('add', v, mask=keep, dtype='f32')

        # Lane 0 only, so the rest of the output buffer keeps its sentinel
        # and the mask can be checked rather than assumed.
        first = b.vcmp('lt', idx, b.splat(b.const(1)))
        b.store(p_out, b.splat(total, dtype='f32'), mask=first)
        b.exit()
    return build


ORACLES = {
    'C01': lambda xs, ys, n, alpha: [xs[i] + ys[i] for i in range(n)],
    'C02': lambda xs, ys, n, alpha: [xs[i] * alpha + ys[i]
                                     for i in range(n)],
    # ISA sec 8.7 fixes the reduction order as a balanced binary tree.
    # These inputs are small exact integers in binary32, so every order
    # gives the same value and a disagreement is not about association.
    'C06': lambda xs, ys, n, alpha: [float(sum(xs[:n]))],
}


def gemm_oracle(a, bm, m, n, k):
    """C = A x B, accumulated in FP32 in increasing k.

    The ISA's contract for the MMA: FP16 inputs, an FP32 accumulator, and
    the k steps taken in order. The oracle does exactly that, so a
    disagreement is the hardware's and not a different summation order.
    """
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc += a[i * k + kk] * bm[kk * n + j]
            row.append(acc)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# C03 / C04 / C05: FP16 GEMM on the tnca-ir input
# ---------------------------------------------------------------------------

def tensormap(base, e0, e1, s0, s1, b0, b1, rank=2, dtype=0, pad=0):
    """The 256-byte tensor map of ISA sec 9D.3, as 32 little-endian words.

    The descriptor extents are the REAL tile, not the fragment: what lies
    past them the transfer engine fills with zero (sec 9E.1 step 7), which
    is what makes an M, N or K tail come out exactly right rather than
    approximately.
    """
    w = [0] * 32
    w[0] = base
    w[1] = e0
    w[2] = e1
    w[6] = s0
    w[7] = s1
    w[11] = (b1 << 32) | b0
    w[13] = (pad << 48) | (dtype << 40) | (rank << 32)
    return w


def pack_words(words):
    return b''.join(struct.pack('<Q', w & ((1 << 64) - 1)) for w in words)


def pack_f16(values):
    return b''.join(struct.pack('<e', v) for v in values)


def gemm_program(m, n, k, *, a_desc, b_desc, out, roles=1):
    """A GEMM for one output tile, emitted into the driver's builder.

    The addresses are baked in, which is why the caller allocates first and
    compiles second. C03-C05 are single-launch cases on the tnca-ir input;
    the dynamic-argument path is C02's and E2E-02's business.
    """
    from tnca_cc.gemm import plan_gemm, build_gemm, check_descriptor_geometry

    def build(b, target):
        plan = plan_gemm(target, m=m, n=n, k=k, dtype='f16')
        check_descriptor_geometry(target, base=a_desc['tile'],
                                  row_stride_bytes=k * 2, elem_bytes=2,
                                  block_cols=plan.frag_k, what='A')
        check_descriptor_geometry(target, base=b_desc['tile'],
                                  row_stride_bytes=n * 2, elem_bytes=2,
                                  block_cols=plan.frag_n, what='B')
        build_gemm(target, plan, name='gemm',
                   a_desc_addr=a_desc['map'], b_desc_addr=b_desc['map'],
                   out_addr=out, builder=b)
        # build_gemm emits the tile and stops; R1's harness appends the
        # exit itself. Without it the core runs off the end of the program
        # and traps -- with the output already correct, which is a
        # confusing way to fail.
        b.exit()
    return build
