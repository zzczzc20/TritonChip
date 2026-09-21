#!/usr/bin/env python3
"""The same kernel at every optimisation level, measured and checked.

    python tutorial/examples/02_opt_levels.py

Two things worth taking away. The first is that O0 emits byte-identical
code to the default: it is the compatibility path, not a level that does
a little. The second is that fewer instructions is not the same as fewer
cycles, and this prints both so you can see where they part company.
"""
import struct
import sys

import triton
import triton.language as tl

from tnca import aot, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
MODES = [('B0 (default)', {}),
         ('O0', {'opt_level': 0}),
         ('O1', {'opt_level': 1}),
         ('O2', {'opt_level': 2})]


@triton.jit
def masked_axpy(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * alpha + y, mask=mask)


def f32(v):
    return b''.join(struct.pack('<f', f) for f in v)


def unf32(b):
    return [struct.unpack_from('<f', b, i * 4)[0] for i in range(len(b) // 4)]


def build(opts):
    return aot.compile_aot(
        masked_axpy,
        signature={'x_ptr': '*fp32', 'y_ptr': '*fp32', 'o_ptr': '*fp32',
                   'n': 'i32', 'alpha': 'fp32', 'BLOCK': 'constexpr'},
        constexprs={'BLOCK': 16}, profile=PROFILE, options=opts,
        output=f"/tmp/tnca-tutorial-02-{opts.get('opt_level', 'base')}.tnca")


def run(art, xs, ys, alpha, n):
    slots = 32
    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        mod = s.load(art.path)
        x = s.allocate(slots * 4, name='x')
        y = s.allocate(slots * 4, name='y')
        o = s.allocate(slots * 4, name='o')
        s.upload(x, f32(xs))
        s.upload(y, f32(ys))
        s.upload(o, struct.pack('<I', 0x7FC00000) * slots)   # a quiet NaN
        ev = s.submit(mod, {'x_ptr': x, 'y_ptr': y, 'o_ptr': o, 'n': n,
                            'alpha': alpha}, grid=(2, 1, 1))
        s.wait(ev)
        got = unf32(s.download(o, slots * 4))[:n]
    return got, ev.device_cycles, ev.state


def main():
    device.prepare(PROFILE, verbose=False)
    n, alpha = 17, 2.5                 # 17 with BLOCK 16: a real tail
    xs = [float(i) - 8 for i in range(32)]
    ys = [float(i) * 0.5 for i in range(32)]
    want = [xs[i] * alpha + ys[i] for i in range(n)]

    print(f"{'mode':<14s} {'words':>6s} {'cycles':>7s} {'vs B0':>7s}  "
          f"{'code == B0':<11s} correct")
    base_words = base_cycles = base_hex = None
    rows = []
    for label, opts in MODES:
        art = build(opts)
        got, cycles, state = run(art, xs, ys, alpha, n)
        words = len(art.asm['hex'].splitlines())
        if base_words is None:
            base_words, base_cycles, base_hex = words, cycles, art.asm['hex']
        same = 'yes' if art.asm['hex'] == base_hex else 'no'
        delta = base_cycles - cycles
        ok = got == want and state == 'SUCCEEDED'
        rows.append((label, words, cycles, delta, same, ok))
        print(f"{label:<14s} {words:>6d} {cycles:>7d} "
              f"{delta:>+7d}  {same:<11s} {'yes' if ok else 'NO'}")

    print()
    print("what to notice:")
    o0 = next(r for r in rows if r[0] == 'O0')
    print(f"  O0 emits the same bytes as the default ({o0[4]}). It is the "
          f"compatibility path:")
    print(f"  if it differed, a program compiled at O0 would not be the "
          f"baseline any more.")
    o2 = next(r for r in rows if r[0] == 'O2')
    print(f"  O2 is {base_words - o2[1]} word(s) smaller and "
          f"{o2[3]} cycle(s) faster. Those two numbers are not the same")
    print(f"  measurement: an instruction removed from a cycle the core "
          f"was already stalled in")
    print(f"  removes nothing.")
    return 0 if all(r[5] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
