#!/usr/bin/env python3
"""Where O2 earns its keep: two layout views that cancel.

    python tutorial/examples/03_layout_views.py

This one is written against the TNCA IR builder rather than Triton,
because a layout view is a TNCA concept. The program reads a 4x4 tile as
rows and columns, reads it back the way it came, and doubles it.

The two views cancel. Whether anything NOTICES is the whole point, and
you can switch the noticing off with `no_ll` to see what it was worth.
"""
import struct
import sys

from tnca import aot, device
from tnca.runtime import Session

PROFILE = 'core-vrf-sync-v1'
ROWS = COLS = 4
N = ROWS * COLS

# A layout is given the way upstream LinearLayout takes it: for each
# input dimension, one basis row per bit of its index, saying where that
# bit sends the value.
TRANSPOSE = [('reg', [[1, 0], [2, 0], [0, 1], [0, 2]])]
BACK = [('row', [[1], [2]]), ('col', [[4], [8]])]


def program(b, target):
    """Read, view, view back, double, store."""
    base = b.special('arg_base')
    p_in = b.sload(base, 0, 'd')
    p_out = b.sload(base, 8, 'd')

    idx = b.arange()
    keep = b.vcmp('lt', idx, b.splat(b.const(N)))
    v = b.load(p_in, mask=keep, other=0, dtype='f32')

    t = b.view(v, TRANSPOSE, ['row', 'col'], [ROWS, COLS],
               what='row-major as column-major')
    u = b.view(t, BACK, ['reg'], [N], what='and back again')

    b.store(p_out, b.vbin('add', u, u, dtype='f32'), mask=keep)
    b.exit()


def f32(v):
    return b''.join(struct.pack('<f', f) for f in v)


def unf32(b):
    return [struct.unpack_from('<f', b, i * 4)[0] for i in range(len(b) // 4)]


def build(opts, tag):
    # A TNCA IR program still declares its parameters: the signature is
    # what lets the runtime put your buffers where the program reads
    # them, and what it checks a launch against.
    return aot.compile_aot(program, kernel_name='views',
                           signature={'in_ptr': '*fp32',
                                      'out_ptr': '*fp32'},
                           source_kind='tnca-ir', profile=PROFILE,
                           options=opts,
                           output=f'/tmp/tnca-tutorial-03-{tag}.tnca')


def run(art, xs):
    with Session(profile=PROFILE, deadline_cycles=50_000) as s:
        mod = s.load(art.path)
        src = s.allocate(N * 4, name='in')
        dst = s.allocate(N * 4, name='out')
        s.upload(src, f32(xs))
        s.upload(dst, struct.pack('<I', 0x7FC00000) * N)
        ev = s.submit(mod, {'in_ptr': src, 'out_ptr': dst}, grid=(1, 1, 1))
        s.wait(ev)
        return unf32(s.download(dst, N * 4)), ev.device_cycles, ev.state


def main():
    device.prepare(PROFILE, verbose=False)
    xs = [float(i) - 8 for i in range(N)]
    want = [x + x for x in xs]

    modes = [('B0 (default)', {}, 'b0'),
             ('O2', {'opt_level': 2}, 'o2'),
             ('O2 with no_ll', {'opt_level': 2, 'no_ll': True}, 'nollv')]
    print(f"{'mode':<16s} {'words':>6s} {'cycles':>7s}  correct")
    out = {}
    for label, opts, tag in modes:
        art = build(opts, tag)
        got, cycles, state = run(art, xs)
        words = len(art.asm['hex'].splitlines())
        out[label] = (words, cycles, got == want and state == 'SUCCEEDED')
        print(f"{label:<16s} {words:>6d} {cycles:>7d}  "
              f"{'yes' if out[label][2] else 'NO'}")

    b0, o2, no = out['B0 (default)'], out['O2'], out['O2 with no_ll']
    print()
    print("what happened:")
    print(f"  At O2 the compiler composes the two views with the real "
          f"upstream LinearLayout,")
    print(f"  finds the composition is the identity over all {N} "
          f"positions, and emits nothing")
    print(f"  for them: {b0[0]} words becomes {o2[0]}, and {b0[1]} cycles "
          f"becomes {o2[1]}.")
    print()
    print(f"  `no_ll` switches the layout ALGEBRA off and keeps everything "
          f"else. Each view is")
    print(f"  then materialised -- an index vector built from the lane "
          f"index and a v.permute --")
    print(f"  which costs {no[0]} words and {no[1]} cycles. That "
          f"difference, {no[1] - o2[1]} cycles, is")
    print(f"  what the layout reasoning is worth here, and it is measured "
          f"rather than asserted.")
    print()
    print("  All three give the same numbers. The views cancel "
          "mathematically whether or not")
    print("  anything notices, which is exactly why the cycle difference "
          "is attributable.")
    return 0 if all(v[2] for v in out.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
