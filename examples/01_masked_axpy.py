#!/usr/bin/env python3
"""
Dynamic masked AXPY -- R2-12 OBS-04, the first of the runnable examples.

Everything here goes through the installed product: `tnca.aot` compiles,
`tnca.runtime.Session` runs. Nothing imports a test module, and the only
thing the kernel is told at compile time is the block size. The addresses,
the length and the scalar arrive at each launch.

    $ python examples/01_masked_axpy.py

`N` is deliberately not a multiple of `BLOCK`: the tail is masked, and the
bytes past the end of the output are checked to be untouched.
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
N = 37                  # 2 full blocks and a 5-element tail


@triton.jit
def masked_axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=mask)


def main():
    art = aot.compile_aot(
        masked_axpy,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': BLOCK},
        profile='core-sim',
        output=os.path.join(tempfile.mkdtemp(prefix='tnca-ex-'),
                            'masked_axpy.tnca'))
    print(f"compiled  {art.path}")
    r = art.report.get('resources', {})
    print(f"          {r.get('instructions')} instructions, "
          f"{r.get('vector_regs_used')} of {r.get('limits', {}).get('vector_regs')} "
          f"vector registers, scratch {r.get('scratch_bytes')}")

    grid_x = (N + BLOCK - 1) // BLOCK
    sentinel = 0x7FC00000          # a NaN pattern, so an untouched tail shows

    with Session(profile='core-sim') as s:
        mod = s.load(art.path)
        # One element of slack either side of the output, to show the tail
        # of the last block writes nothing past N.
        x = s.allocate(grid_x * BLOCK * 4, name='x')
        y = s.allocate(grid_x * BLOCK * 4, name='y')
        o = s.allocate(grid_x * BLOCK * 4, name='o')

        xs = [float(i) for i in range(grid_x * BLOCK)]
        ys = [100.0 + i for i in range(grid_x * BLOCK)]
        s.upload(x, b''.join(struct.pack('<f', v) for v in xs))
        s.upload(y, b''.join(struct.pack('<f', v) for v in ys))
        s.upload(o, struct.pack('<I', sentinel) * (grid_x * BLOCK))

        # The same artifact, twice, with a different scalar each time.
        for alpha in (2.0, -0.5):
            ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                                'n': N, 'alpha': alpha},
                          grid=(grid_x, 1, 1))
            s.wait(ev)
            raw = s.download(o, grid_x * BLOCK * 4)
            got = list(struct.unpack(f'<{grid_x * BLOCK}f', raw))
            want = [xs[i] * alpha + ys[i] for i in range(N)]
            body_ok = got[:N] == want
            tail = struct.unpack(f'<{grid_x * BLOCK - N}I', raw[N * 4:])
            tail_ok = all(v == sentinel for v in tail)
            print(f"alpha={alpha:<5}  first {N} correct: {body_ok}   "
                  f"{len(tail)} masked element(s) untouched: {tail_ok}")
            if not (body_ok and tail_ok):
                return 1
    print("ok")
    return 0


if __name__ == '__main__':
    sys.exit(main())
