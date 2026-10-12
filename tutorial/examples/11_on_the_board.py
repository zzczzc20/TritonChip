#!/usr/bin/env python3
"""One artifact, run on whichever backends you name: the simulator, the
board's simulated twin, and the U280 itself.

    # anywhere with iverilog: the simulator and the board's register protocol on RTL
    python tutorial/examples/11_on_the_board.py --backend sim-file fpga-u50-sim

    # on the host with the card, image already loaded (flow/fpga/program.sh --go)
    python tutorial/examples/11_on_the_board.py --backend fpga-u280 --xclbin tnca_u280.xclbin

The artifact is compiled once, for a profile; the backend is only how this
host reaches a device, so the same file runs on all of them. The cycle
counts are the device's own, and they must agree: the board runs the RTL the
simulators run.
"""
import argparse
import struct
import sys
import time

import triton
import triton.language as tl

from tnca import aot, backend, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
N, BLOCK = 1000, 16


@triton.jit
def axpy(x_ptr, y_ptr, o_ptr, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, alpha * x + y, mask=mask)


def f32(values):
    return b''.join(struct.pack('<f', v) for v in values)


def unf32(blob):
    return [struct.unpack_from('<f', blob, i * 4)[0] for i in range(len(blob) // 4)]


def run(art, name, xclbin):
    kw = {}
    if backend.needs_elaborator(name):
        device.prepare(PROFILE, verbose=False)
    if backend.needs_hardware(name):
        kw['xclbin'] = xclbin
    xs = [float(i % 97) for i in range(N)]
    ys = [float(i % 13) - 6.0 for i in range(N)]
    alpha = 0.5
    with Session(profile=PROFILE, backend=name, deadline_cycles=2_000_000, **kw) as s:
        mod = s.load(art.path)
        x, y, o = (s.allocate(N * 4, name=k) for k in 'xyo')
        s.upload(x, f32(xs))
        s.upload(y, f32(ys))
        grid = ((N + BLOCK - 1) // BLOCK, 1, 1)
        t0 = time.perf_counter()
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o,
                            'alpha': alpha, 'n': N}, grid=grid)
        s.wait(ev)
        wall = time.perf_counter() - t0
        got = unf32(s.download(o, N * 4))
    want = [struct.unpack('<f', struct.pack('<f', alpha * a + b))[0]
            for a, b in zip(xs, ys)]
    ok = ev.state == 'SUCCEEDED' and got == want
    print(f"  {name:<13s} {ev.state:<10s} {ev.device_cycles:>7d} cycle(s)  "
          f"host wall {wall:7.3f} s  correct={got == want}")
    return ok, ev.device_cycles


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', nargs='+', default=['sim-file'])
    ap.add_argument('--xclbin', help='the image the card is running (board backends only)')
    args = ap.parse_args()

    art = aot.compile_aot(
        axpy,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'alpha': 'fp32', 'n': 'i32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': BLOCK}, profile=PROFILE,
        options={'opt_level': 2}, output='/tmp/tnca-tutorial-11.tnca')
    print(f"one artifact, key {art.cache_key[:12]}: AXPY over {N} elements, "
          f"{(N + BLOCK - 1) // BLOCK} programs")
    results = [run(art, name, args.xclbin) for name in args.backend]
    ok = all(r[0] for r in results)
    cycles = {r[1] for r in results}
    if len(results) > 1:
        print(f"  the same cycle count on every backend: {len(cycles) == 1}")
        ok &= len(cycles) == 1
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
