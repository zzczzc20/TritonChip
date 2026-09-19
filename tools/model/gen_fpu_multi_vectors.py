#!/usr/bin/env python3
"""
Vectors for tb/xls/tb_fpu_multi.v -- multi-precision FMA.

The expected value is computed the way the ISA specifies the operation
(sec 9C.1): widen both operands exactly, form the product exactly, add the
binary32 accumulator, round once. tools/model/fp32.fma_ref does the last
three steps in Fractions; floatfmt.decode does the widening, exactly, because
every one of these formats is a subset of binary32.

That the reference widens with the *oracle* rather than with the hardware's
widening unit is deliberate. If both used the same converter, a converter bug
would cancel out and the test would pass on a wrong answer.
"""
import argparse
import os
import random
import struct
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from floatfmt import CODE, FORMATS, NAN, decode, width  # noqa: E402
from fp32 import f2b, fma_ref  # noqa: E402

FORMATS_USED = ['f32', 'bf16', 'f16', 'e5m2', 'e4m3']


def to_f32_bits(fmt, bits):
    """Exact widening, from the oracle rather than from the hardware."""
    v = decode(fmt, bits)
    if v == NAN:
        return 0x7fc00000
    if isinstance(v, tuple):
        return (v[1] << 31) | (0xff << 23)
    if v == 0:
        sign = (bits >> (width(fmt) - 1)) & 1
        return sign << 31
    return f2b(float(v)) if abs(v) != 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-n', '--per-format', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for fmt in FORMATS_USED:
        w = width(fmt)
        pool = (list(range(1 << w)) if w <= 8
                else [rng.getrandbits(w) for _ in range(args.per_format)])
        for a in pool:
            b = rng.getrandbits(w)
            c = rng.getrandbits(32)
            if ((c >> 23) & 0xff) == 0xff:
                c &= 0x7f7fffff              # keep the addend finite
            af, bf = to_f32_bits(fmt, a), to_f32_bits(fmt, b)
            want = fma_ref(af, bf, c)
            rows.append((CODE[fmt], a, b, c, want))

    with open(args.output, 'w') as fh:
        for f, a, b, c, y in rows:
            fh.write(f"{f} {a:08x} {b:08x} {c:08x} {y:08x}\n")
    print(f"{len(rows)} vectors -> {args.output} "
          f"({len(FORMATS_USED)} formats; FP8 exhaustive in a)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
