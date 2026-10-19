#!/usr/bin/env python3
"""
Vectors for tb/core/tb_vector_fpu_f2i.v: binary32 -> int32/uint32 against the
ISA sec 8.4 text (tools/model/f2i.py).

Two input sets:
  * the SystemC numeric validator's 21,614 floats (14 specials, 20,000 random
    words, 1,600 in exponents 150..157), regenerated with the same seed and
    the same RNG call order as tools/performance_harness/systemc/
    validate_numeric.py, so the 172,912 (signed, mode) cases it found
    divergent on the old RTL are all included;
  * a directed set at every boundary the conversion has: each exponent
    120..160 with edge mantissas, ties, +-2^31, 2^32 - ulp, (-1, 0), (0, 1),
    subnormals, zeros, infinities and NaNs.

Each line: "<signed> <mode> <raw:08x> <expected:08x>". Lines are grouped so
that 16 consecutive lines share (signed, mode) and form one vector operation.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f2i import f32_to_int, MODES          # noqa: E402

VLEN = 16


def systemc_floats():
    rng = random.Random(7319)
    # validate_numeric.py draws its 20,000 multiplier pairs first.
    for _ in range(20000):
        rng.getrandbits(64)
        rng.getrandbits(64)
    floats = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
              0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x40200000,
              0x4f000000, 0xcf000000, 0x4f800000, 0xcf800000]
    floats += [rng.getrandbits(32) for _ in range(20000)]
    floats += [(e << 23) | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
               for e in range(150, 158) for _ in range(200)]
    return floats


def directed_floats():
    mants = [0, 1, 2, 3, 0x3fffff, 0x400000, 0x400001, 0x5fffff, 0x600000,
             0x7ffffe, 0x7fffff, 0x2aaaaa, 0x555555]
    out = []
    for e in range(120, 161):
        for m in mants:
            for s in (0, 1):
                out.append((s << 31) | (e << 23) | m)
    out += [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
            0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff812345,
            0x7f7fffff, 0xff7fffff]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()
    sets = [systemc_floats(), directed_floats()]
    lines = []
    for floats in sets:
        pad = (-len(floats)) % VLEN
        floats = floats + floats[:pad]
        for signed in (0, 1):
            for mode in MODES:
                for raw in floats:
                    lines.append(f"{signed} {mode} {raw:08x} {f32_to_int(raw, signed, mode):08x}")
    with open(a.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} vectors ({len(sets[0])} SystemC floats + {len(sets[1])} directed) x 8 -> {a.out}")


if __name__ == '__main__':
    main()
