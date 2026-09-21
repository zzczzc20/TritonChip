#!/usr/bin/env python3
"""The smallest whole thing: a Triton kernel, compiled, run, checked.

    python tutorial/examples/01_hello_vector_add.py

Nothing here is a test helper. Every call is one a user makes.
"""
import struct
import sys

import triton
import triton.language as tl

from tnca import aot, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'


@triton.jit
def vector_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x + y, mask=mask)


def f32(values):
    """Python floats to the bytes the device reads."""
    return b''.join(struct.pack('<f', v) for v in values)


def unf32(blob):
    return [struct.unpack_from('<f', blob, i * 4)[0]
            for i in range(len(blob) // 4)]


def main():
    n = 16

    # 1. Build the device once. `prepare` compiles the simulator if it is
    #    not already built; `Session` will not build one behind your back.
    device.prepare(PROFILE, verbose=False)

    # 2. Compile. The signature says what the parameters ARE, which is
    #    separate from what they will be at launch -- one artifact runs
    #    with any arguments of that shape.
    art = aot.compile_aot(
        vector_add,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': n},
        profile=PROFILE,
        output='/tmp/tnca-tutorial-01.tnca')
    print(f"compiled  {len(art.asm['hex'].splitlines())} machine word(s), "
          f"cache key {art.cache_key[:12]}")

    # 3. Run. A session owns the device; buffers and events belong to it.
    xs = [float(i) for i in range(n)]
    ys = [float(100 - i) for i in range(n)]
    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        mod = s.load(art.path)
        x = s.allocate(n * 4, name='x')
        y = s.allocate(n * 4, name='y')
        o = s.allocate(n * 4, name='o')
        s.upload(x, f32(xs))
        s.upload(y, f32(ys))
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n},
                      grid=(1, 1, 1))
        s.wait(ev)
        got = unf32(s.download(o, n * 4))
        cycles = ev.device_cycles

    # 4. Check against an oracle computed here, not by the device.
    want = [xs[i] + ys[i] for i in range(n)]
    print(f"launched  {ev.state}, {cycles} device cycle(s)")
    print(f"result    {got[:4]} ...")
    print(f"expected  {want[:4]} ...")
    if got != want:
        print("MISMATCH")
        return 1
    print("correct   every element matches")
    return 0


if __name__ == '__main__':
    sys.exit(main())
