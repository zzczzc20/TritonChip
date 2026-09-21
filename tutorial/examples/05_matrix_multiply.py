#!/usr/bin/env python3
"""A 16x16x16 FP16 matrix multiply on the matrix unit.

    python tutorial/examples/05_matrix_multiply.py

This uses the compiler's GEMM planner rather than a hand-written kernel,
because the interesting part is the machinery underneath: tensor
descriptors, an asynchronous transfer into a register window, the MMA,
and reading the accumulator back out a row at a time.

Note the order. The addresses are baked into the program, so the buffers
have to exist BEFORE the code does. That is the opposite of the
elementwise examples and is why this one allocates first.
"""
import struct
import sys

from tnca import aot, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
M = N = K = 16


def f16_bits(x):
    """One float as its binary16 bit pattern."""
    return struct.unpack('<H', struct.pack('<e', x))[0]


def pack_f16(values):
    return b''.join(struct.pack('<H', f16_bits(v)) for v in values)


def unf32(blob):
    return [struct.unpack_from('<f', blob, i * 4)[0]
            for i in range(len(blob) // 4)]


def main():
    # The planner and the builder come from the compiler package.
    from tnca_cc.gemm import plan_gemm, build_gemm
    sys.path.insert(0, 'tools/r3v2')
    import workloads as W          # its tensormap helper and oracle

    device.prepare(PROFILE, verbose=False)

    a_vals = W.inputs(M * K, seed=42, lo=-4, hi=4)
    b_vals = W.inputs(K * N, seed=43, lo=-4, hi=4)
    want = W.gemm_oracle(a_vals, b_vals, M, N, K)

    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        # Allocate first: the program will name these addresses.
        a_map = s.allocate(256, name='descriptor for A')
        b_map = s.allocate(256, name='descriptor for B')
        a_tile = s.allocate(M * K * 2, name='A')
        b_tile = s.allocate(K * N * 2, name='B')
        out = s.allocate((M * N + N) * 4, name='C')

        def program(b, target):
            plan = plan_gemm(target, m=M, n=N, k=K, dtype='f16')
            build_gemm(target, plan, name='gemm',
                       a_desc_addr=a_map.addr, b_desc_addr=b_map.addr,
                       out_addr=out.addr, builder=b)
            b.exit()

        art = aot.compile_aot(program, kernel_name='gemm', signature={},
                              source_kind='tnca-ir', profile=PROFILE,
                              options={'opt_level': 2},
                              output='/tmp/tnca-tutorial-05.tnca')
        mod = s.load(art.path)

        s.upload(a_tile, pack_f16(a_vals))
        s.upload(b_tile, pack_f16(b_vals))
        s.upload(a_map, W.pack_words(
            W.tensormap(a_tile.addr, M, K, K * 2, 2, 16, 16, dtype=0)))
        s.upload(b_map, W.pack_words(
            W.tensormap(b_tile.addr, K, N, N * 2, 2, 16, 16, dtype=0)))
        # One row past the tile, all sentinel: a store that leaked would
        # still satisfy every value check without this.
        s.upload(out, struct.pack('<I', 0x7FC00000) * (M * N + N))

        ev = s.submit(mod, {})
        s.wait(ev)
        raw = s.download(out, (M * N + N) * 4)

    got = unf32(raw)
    flat = [v for row in want for v in row]
    body = got[:M * N] == flat
    tail = struct.unpack(f'<{N}I', raw[M * N * 4:])
    tail_ok = all(v == 0x7FC00000 for v in tail)

    print(f"compiled  {len(art.asm['hex'].splitlines())} machine word(s) "
          f"for a {M}x{N}x{K} FP16 GEMM")
    print(f"launched  {ev.state}, {ev.device_cycles} device cycle(s)")
    print(f"values    {'correct' if body else 'WRONG'} against a "
          f"double-precision oracle")
    print(f"tail      {len(tail)} sentinel slot(s) "
          f"{'intact' if tail_ok else 'TOUCHED'}")
    print(f"row 0     {[round(v, 2) for v in got[:4]]} ...")
    print(f"expected  {[round(v, 2) for v in flat[:4]]} ...")
    if not (body and tail_ok):
        return 1
    print()
    print("the shape of what ran, from the assembly's own header:")
    for line in art.asm['asm'].splitlines()[:9]:
        print(f"    {line}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
