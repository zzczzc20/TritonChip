#!/usr/bin/env python3
"""A matrix multiply written as a Triton kernel.

    python tutorial/examples/08_triton_gemm.py

[05] used the compiler's GEMM planner, which drives the matrix unit.
This page writes the same thing as a @triton.jit kernel, twice:

  * by hand, broadcasting a scalar of A across a row of B, which runs
    on the vector unit and is the version you can modify a line of and
    watch change;
  * with `tl.dot`, which lowers onto `m.mma.async` -- one instruction
    for the whole 16x16x16 product.

It also shows where this frontend's support ends, with the exact
errors, because knowing the boundary is more useful than discovering
it.
"""
import struct
import sys

import triton
import triton.language as tl

from tnca import aot, device
from tnca.errors import TncaError
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
M = N = K = 16


@triton.jit
def gemm_row_per_program(a_ptr, b_ptr, c_ptr,
                         K: tl.constexpr, N: tl.constexpr,
                         BLOCK: tl.constexpr):
    """One program computes one row of C.

        C[m, :] = sum over k of A[m, k] * B[k, :]

    Each lane holds one column, so a row of B is exactly one vector and
    A[m, k] is a scalar broadcast across it. `tl.static_range` unrolls
    the k loop while Triton traces, so what reaches the backend is
    straight-line code with no control flow in it at all.
    """
    m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK)

    acc = tl.load(a_ptr + m * K + 0) * tl.load(b_ptr + 0 * N + offs_n)
    for k in tl.static_range(1, K):
        acc = acc + (tl.load(a_ptr + m * K + k) *
                     tl.load(b_ptr + k * N + offs_n))

    tl.store(c_ptr + m * N + offs_n, acc)


@triton.jit
def gemm_dot(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
             M: tl.constexpr):
    """The same product, as one `tl.dot`.

    Nothing here is written for this machine. The 2-D index arithmetic
    is the shape every Triton matmul tutorial has; what the backend
    does with it is recognise that `om[:, None] * K + ok[None, :]`
    describes a tile, build a TensorMap for it, and fetch it with the
    transfer engine -- because a packed FP16 fragment is not something
    vector loads can assemble (ISA sec 9C.2, and sec 8.4 reserves the
    pack that would be needed).

    The product itself is one `m.mma.async`, and the result leaves the
    accumulator a row at a time through the vector unit (sec 9B).
    """
    om = tl.arange(0, M)
    ok = tl.arange(0, K)
    on = tl.arange(0, N)
    a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
    b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
    tl.store(c_ptr + om[:, None] * N + on[None, :], tl.dot(a, b))


def _reason(e):
    """The one line of a compiler error that says why.

    A Triton frontend error echoes the source with a caret under the
    offending expression and puts the reason on the line after it; ours
    puts it first. Both bury it in something too long to read here.
    """
    lines = [ln.rstrip() for ln in str(e).splitlines()]
    for i, ln in enumerate(lines[:-1]):
        if set(ln.strip()) == {'^'}:
            return lines[i + 1].strip()[:70]
    return lines[0].strip()[:70]


def f32(v):
    return b''.join(struct.pack('<f', x) for x in v)


def unf32(b):
    return [struct.unpack_from('<f', b, i * 4)[0] for i in range(len(b) // 4)]


def oracle(a, b):
    """C = A @ B, accumulated in double and rounded once per element."""
    out = []
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += a[m * K + k] * b[k * N + n]
            out.append(struct.unpack('<f', struct.pack('<f', acc))[0])
    return out


def main():
    device.prepare(PROFILE, verbose=False)
    ok = True

    art = aot.compile_aot(
        gemm_row_per_program,
        signature={'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32',
                   'K': 'constexpr', 'N': 'constexpr',
                   'BLOCK': 'constexpr'},
        constexprs={'K': K, 'N': N, 'BLOCK': N},
        profile=PROFILE, options={'opt_level': 2},
        output='/tmp/tnca-tutorial-08.tnca')
    print(f"compiled  {len(art.asm['hex'].splitlines())} machine word(s) "
          f"for a {M}x{N}x{K} FP32 GEMM, written in Triton")

    a = [float((i % 7) - 3) for i in range(M * K)]
    b = [float((i % 5) - 2) for i in range(K * N)]
    want = oracle(a, b)

    with Session(profile=PROFILE, deadline_cycles=200_000) as s:
        mod = s.load(art.path)
        da = s.allocate(M * K * 4, name='A')
        db = s.allocate(K * N * 4, name='B')
        dc = s.allocate(M * N * 4, name='C')
        s.upload(da, f32(a))
        s.upload(db, f32(b))
        s.upload(dc, struct.pack('<I', 0x7FC00000) * (M * N))
        # One program per row of C: the grid is the M dimension.
        ev = s.submit(mod, {'a_ptr': da, 'b_ptr': db, 'c_ptr': dc},
                      grid=(M, 1, 1))
        s.wait(ev)
        got = unf32(s.download(dc, M * N * 4))

    print(f"launched  {ev.state}, grid of {M} program(s), "
          f"{ev.device_cycles} device cycle(s)")
    print(f"values    {'correct' if got == want else 'WRONG'} against a "
          f"double-precision oracle")
    print(f"C[0,:4]   {got[:4]}")
    print(f"expected  {want[:4]}")
    ok &= got == want

    if got != want:
        bad = next(i for i in range(M * N) if got[i] != want[i])
        print(f"          first difference at element {bad}: "
              f"{got[bad]} != {want[bad]}")

    # ---- the same product, as one instruction --------------------------
    dot = aot.compile_aot(
        gemm_dot,
        signature={'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': '*fp32',
                   'K': 'constexpr', 'N': 'constexpr', 'M': 'constexpr'},
        constexprs={'K': K, 'N': N, 'M': M},
        profile=PROFILE, options={'opt_level': 2},
        output='/tmp/tnca-tutorial-08-dot.tnca')
    asm = dot.asm['asm']
    print()
    print(f"compiled  {len(dot.asm['hex'].splitlines())} machine word(s) "
          f"for the SAME product written as tl.dot")
    for want_mnem, what in (('td.load', 'the TensorMap snapshot'),
                            ('t.ld.async', 'the tile fetch'),
                            ('m.mma.async', 'the 16x16x16 product'),
                            ('a.vread', 'the accumulator epilogue')):
        n = sum(1 for ln in asm.splitlines()
                if ln.strip().startswith(want_mnem))
        print(f"  {want_mnem:<14s} x{n:<3d} {what}")
    print(f"  scratch       {dot.manifest['resources']['scratch_bytes']} "
          f"byte(s) for the descriptors it writes")

    print()
    print("where this frontend's support currently ends")
    print("(these are the real errors, not hypotheticals):")
    print()

    @triton.jit
    def dot_f32(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
                BLOCK: tl.constexpr):
        offs_m = tl.arange(0, BLOCK)
        offs_n = tl.arange(0, BLOCK)
        offs_k = tl.arange(0, BLOCK)
        aa = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        bb = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
                 tl.dot(aa, bb))

    @triton.jit
    def runtime_loop(x_ptr, o_ptr, K, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        acc = tl.load(x_ptr + offs)
        for _ in range(0, K):          # K is NOT a constexpr
            acc = acc + acc
        tl.store(o_ptr + offs, acc)

    for label, fn, sig, ce in (
            ('tl.dot on f32', dot_f32,
             {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32',
              'K': 'constexpr', 'N': 'constexpr', 'BLOCK': 'constexpr'},
             {'K': 16, 'N': 16, 'BLOCK': 16}),
            ('a loop with a runtime bound', runtime_loop,
             {'x_ptr': '*fp32', 'o_ptr': '*fp32', 'K': 'i32',
              'BLOCK': 'constexpr'}, {'BLOCK': 16})):
        try:
            aot.compile_aot(fn, signature=sig, constexprs=ce,
                            profile=PROFILE,
                            output=f'/tmp/tnca-tut-08-{label[:6]}.tnca')
            print(f"  {label:<28s} compiled -- this note is out of date")
        except TncaError as e:
            print(f"  {label:<28s} [{getattr(e, 'code', '?')}]")
            print(f"  {'':<28s} {_reason(e)}")

    print()
    print("  `tl.dot` itself is supported -- the fp16 kernel above is "
          "one, and it")
    print("  becomes a single m.mma.async. What is refused is a dot "
          "with binary32")
    print("  inputs, because ISA sec 9C's mod[3:2] names FP16, BF16 and "
          "the two")
    print("  FP8 encodings and reserves the rest: there is no f32 "
          "matrix unit to")
    print("  lower it onto, and widening the operands behind your back "
          "would be")
    print("  a different computation.")
    print()
    print("  A loop whose bound is a runtime value stays an scf.for, "
          "which has no")
    print("  TNCA lowering. That is why the kernels here use "
          "tl.static_range: it")
    print("  is unrolled before the backend ever sees it.")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
