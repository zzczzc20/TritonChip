#!/usr/bin/env python3
"""Generate differential test vectors for tnca_fp32_fma."""
import argparse
import random
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32 import fma_ref

F = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]

# Values chosen to sit on the boundaries the RTL's alignment and normalisation
# paths actually branch on.
CORNERS = [
    0x00000000, 0x80000000,                     # +-0
    0x00000001, 0x80000001,                     # +-min subnormal
    0x007fffff, 0x807fffff,                     # +-max subnormal
    0x00800000, 0x80800000,                     # +-min normal
    0x7f7fffff, 0xff7fffff,                     # +-max normal
    0x7f800000, 0xff800000,                     # +-inf
    0x7fc00000, 0x7f800001, 0xffc00001,         # NaNs
    F(1.0), F(-1.0), F(2.0), F(0.5), F(3.0),
    F(1.0000001), F(0.9999999),
    0x33800000,                                 # 2^-24
    0x4b800000,                                 # 2^24
    0x7f000000, 0x00ffffff,
]


def rand_bits(rng, bias):
    """bias picks how exponents are spread, which is what stresses alignment."""
    if bias == 'corner':
        return rng.choice(CORNERS)
    s = rng.getrandbits(1) << 31
    f = rng.getrandbits(23)
    if bias == 'near':
        e = rng.randint(110, 145)
    elif bias == 'wide':
        e = rng.randint(1, 254)
    elif bias == 'sub':
        e = rng.choice([0, 0, 1, 2])
    else:
        return rng.getrandbits(32)
    return s | (e << 23) | f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-n', '--count', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=20260918)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    biases = ['corner', 'near', 'wide', 'sub', 'any']

    lines = []

    # Exhaustive over the corner cross-product for fma, mul and sub.
    for a in CORNERS:
        for b in CORNERS:
            for c in (0x00000000, 0x80000000, F(1.0), F(-1.0), 0x7f800000, 0x00000001):
                for ctrl in (0, 1, 2, 4):
                    exp = fma_ref(a, b, c, sub_c=bool(ctrl & 1),
                                  mul_only=bool(ctrl & 2), ftz=bool(ctrl & 4))
                    lines.append(f"{a:08x} {b:08x} {c:08x} {ctrl:x} {exp:08x}")

    for _ in range(args.count):
        ba, bb, bc = (rng.choice(biases) for _ in range(3))
        a, b, c = rand_bits(rng, ba), rand_bits(rng, bb), rand_bits(rng, bc)
        ctrl = rng.choice([0, 0, 0, 1, 2, 4, 5])
        exp = fma_ref(a, b, c, sub_c=bool(ctrl & 1),
                      mul_only=bool(ctrl & 2), ftz=bool(ctrl & 4))
        lines.append(f"{a:08x} {b:08x} {c:08x} {ctrl:x} {exp:08x}")

    # Cancellation sweep: a*b and c chosen to nearly annihilate.
    for _ in range(args.count // 4):
        e = rng.randint(30, 220)
        m = rng.getrandbits(23)
        p = (e << 23) | m
        delta = rng.randint(0, 40)
        ce = max(1, e - delta)
        cbits = (ce << 23) | rng.getrandbits(23)
        cbits |= 0x80000000
        exp = fma_ref(p, F(1.0), cbits)
        lines.append(f"{p:08x} {F(1.0):08x} {cbits:08x} 0 {exp:08x}")

    with open(args.output, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines)} vectors -> {args.output}")


if __name__ == '__main__':
    main()
