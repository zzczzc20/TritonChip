#!/usr/bin/env python3
"""
A three-program chain over a grid wider than four -- R2-12 OBS-04.

Two things this shows that a single launch does not:

  - the grid is 6 x 2, so `program_id(0)` and `program_id(1)` are the
    logical indices on their own axes, not a linear id folded into x, and
    not the tile number;
  - B and C live in device memory. The host uploads A and downloads D, and
    the two intermediates are never read back. Each launch depends on the
    one before it, and the runtime orders them.

    $ python examples/02_chain_grid.py
"""
import os
import struct
import sys
import tempfile

import triton
import triton.language as tl

from tnca import aot
from tnca.runtime import Session

BLOCK = 16
GRID = (6, 2, 1)
N = BLOCK * GRID[0] * GRID[1]


@triton.jit
def scale_add(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    # The linear program index, rebuilt from the two axes. If program_id
    # returned the linear id on axis 0, this would be wrong for every
    # program with y != 0 -- which is what makes it worth writing this way.
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=mask)


def f32(values):
    return b''.join(struct.pack('<f', v) for v in values)


def main():
    art = aot.compile_aot(
        scale_add,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': BLOCK},
        profile='core-sim',
        output=os.path.join(tempfile.mkdtemp(prefix='tnca-ex-'),
                            'scale_add.tnca'))
    print(f"compiled  {os.path.basename(art.path)}   "
          f"grid {GRID[0]}x{GRID[1]} = {GRID[0] * GRID[1]} programs, "
          f"N={N}")

    with Session(profile='core-sim') as s:
        mod = s.load(art.path)
        a, b, c, d = (s.allocate(N * 4, name=n) for n in 'ABCD')
        zero = s.allocate(N * 4, name='zero')
        src = [float(i) for i in range(N)]
        s.upload(a, f32(src))
        s.upload(zero, f32([0.0] * N))

        #  B = 2A      C = 3B      D = C + A
        e1 = s.submit(mod, {'x_ptr': a, 'y_ptr': zero, 'o_ptr': b,
                            'n': N, 'alpha': 2.0}, grid=GRID)
        e2 = s.submit(mod, {'x_ptr': b, 'y_ptr': zero, 'o_ptr': c,
                            'n': N, 'alpha': 3.0}, grid=GRID,
                      depends_on=[e1])
        e3 = s.submit(mod, {'x_ptr': c, 'y_ptr': a, 'o_ptr': d,
                            'n': N, 'alpha': 1.0}, grid=GRID,
                      depends_on=[e2])
        s.wait(e3)
        print(f"          {e1.state} -> {e2.state} -> {e3.state}, "
              f"0 host reads of B and C")

        got = list(struct.unpack(f'<{N}f', s.download(d, N * 4)))
        want = [6.0 * v + v for v in src]
        if got != want:
            bad = next(i for i in range(N) if got[i] != want[i])
            print(f"WRONG at element {bad}: {got[bad]} != {want[bad]}")
            return 1
        print(f"          D = 6A + A over all {N} elements: correct")
    print("ok")
    return 0


if __name__ == '__main__':
    sys.exit(main())
