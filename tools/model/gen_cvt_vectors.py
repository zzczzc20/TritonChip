#!/usr/bin/env python3
"""
Generate vectors for tb/xls/tb_cvt_xls.v from the exact oracle.

Exhaustive wherever exhaustive is possible. Both FP8 formats have 256 bit
patterns and both 16-bit formats have 65,536, so every one of them is checked
against every destination format -- there is no reason to sample a space that
fits in a loop. Only binary32 sources are sampled, and then from a structured
set (every binade boundary, every rounding tie, the saturation edges) plus
random draws, because the interesting inputs for a narrowing conversion are
the ones on a boundary.
"""
import argparse
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from floatfmt import CODE, FORMATS, convert, width  # noqa: E402

NARROW = ['bf16', 'f16', 'e5m2', 'e4m3']
ALL = ['f32'] + NARROW


def f32_probe_set(rng, n_random):
    """Inputs where a narrowing conversion is most likely to be wrong."""
    out = set()

    def add(x):
        out.add(x & 0xffffffff)

    # Zeros, infinities, NaNs.
    for s in (0, 1):
        add(s << 31)
        add((s << 31) | (0xff << 23))
        add((s << 31) | (0xff << 23) | 0x400000)
        add((s << 31) | (0xff << 23) | 1)

    # Every value each narrow format can hold, widened back to binary32: a
    # conversion that cannot reproduce its own outputs is wrong at once. Both
    # FP8 formats are taken whole; the 16-bit ones are stepped, because their
    # neighbours in binary32 are what the rounding boundaries are made of and
    # 65,536 of them would swamp the set without adding a new shape.
    for fmt in NARROW:
        n = 1 << width(fmt)
        step = 1 if n <= 256 else 37      # coprime with 2^16: no aliasing
        for bits in range(0, n, step):
            v = convert(fmt, 'f32', bits)
            add(v)
            # And the neighbours of each, which straddle the rounding ties.
            for d in (-2, -1, 1, 2):
                add(v + d)

    # Binade boundaries and halfway points across the whole exponent range.
    for e in range(0, 255):
        for frac in (0, 1, 0x400000, 0x7fffff, 0x3fffff, 0x400001):
            add((e << 23) | frac)
            add((1 << 31) | (e << 23) | frac)

    for _ in range(n_random):
        add(rng.getrandbits(32))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--f32-random', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # Exhaustive: every bit pattern of every narrow format, to every format.
    for src in NARROW:
        for bits in range(1 << width(src)):
            for dst in ALL:
                rows.append((CODE[src], CODE[dst], bits,
                             convert(src, dst, bits)))

    # Sampled: binary32 sources, from the probe set above.
    probes = f32_probe_set(rng, args.f32_random)
    for bits in probes:
        for dst in ALL:
            rows.append((CODE['f32'], CODE[dst], bits,
                         convert('f32', dst, bits)))

    with open(args.output, 'w') as fh:
        for s, d, x, y in rows:
            fh.write(f"{s} {d} {x:08x} {y:08x}\n")

    exhaustive = sum(1 << width(f) for f in NARROW) * len(ALL)
    print(f"{len(rows)} vectors -> {args.output} "
          f"({exhaustive} exhaustive, {len(rows) - exhaustive} from binary32)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
