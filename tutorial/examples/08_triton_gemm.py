#!/usr/bin/env python3
"""A matrix multiply written as a Triton kernel.

    python tutorial/examples/08_triton_gemm.py

[05] used the compiler's GEMM planner, which drives the matrix unit.
This one is a @triton.jit kernel you write yourself, and it runs on the
vector unit. It is slower, and it is the one you can modify.

It also shows where this frontend's support currently ends, with the
exact errors, because knowing the boundary is more useful than
discovering it.
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

    print()
    print("where this frontend's support currently ends")
    print("(these are the real errors, not hypotheticals):")
    print()

    @triton.jit
    def uses_dot(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
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
            ('tl.dot', uses_dot,
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
    print("  `tl.dot` is refused by Triton's OWN frontend for this "
          "target, before")
    print("  anything of ours runs: the target publishes no dot "
          "precisions at all,")
    print("  so no argument to input_precision is valid. A loop whose "
          "bound is a")
    print("  runtime value stays an scf.for, whose bounds arrive as "
          "arith.bitcast;")
    print("  neither has a TNCA lowering. That is why the kernel above "
          "uses")
    print("  tl.static_range: it is unrolled before the backend ever "
          "sees it.")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
