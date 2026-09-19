#!/usr/bin/env python3
"""
Generate SFU vectors and the reference the hardware is measured against.

ISA sec 8.8 gives a relative error bound rather than an exact result, so the
vector file carries the *exact* value of each function as a binary32 pair of
bounds: the largest result that is still within tolerance and the smallest.
The testbench then checks membership in that interval rather than equality,
which is what an approximation contract means.

The reference is computed in 60-digit decimal, not host floats. A reference
that itself carried only binary32 precision could not tell a result inside
the 2^-22 band from one outside it.
"""
import argparse
import os
import random
import struct
import sys
from decimal import Decimal, getcontext

getcontext().prec = 60

HERE = os.path.dirname(os.path.abspath(__file__))

TOL = Decimal(2) ** -22          # ISA sec 8.8
OPS = ('exp2', 'log2', 'rcp', 'rsqrt')

PZERO, NZERO = 0x0000_0000, 0x8000_0000
PINF,  NINF  = 0x7f80_0000, 0xff80_0000
ONE          = 0x3f80_0000
ANY_NAN      = 'nan'             # checked by class, not by payload


def specials():
    """ISA sec 8.8's special-value table, as (op, x, expected) triples.

    The bound in sec 8.8 says nothing about these: they are *specified*
    results, so they are checked exactly rather than inside a band. NaN is
    the one exception -- the ISA promises a quiet NaN and not a payload, so
    demanding a particular pattern would be testing an accident.

    Subnormal inputs appear here because sec 8.8 flushes them to a zero of
    the same sign, which means every function sees them as its zero case.
    That has a consequence worth pinning down rather than leaving to be
    discovered: rsqrt of a negative *normal* is a NaN, but rsqrt of a
    negative *subnormal* is -inf, because the flush happens first.
    """
    sub = [0x0000_0001, 0x007f_ffff]                 # smallest, largest
    nsub = [b | 0x8000_0000 for b in sub]
    nans = [0x7fc0_0000, 0x7f80_0001, 0xffc0_0000]   # quiet, signalling, signed
    zeroish_p = [PZERO] + sub
    zeroish_n = [NZERO] + nsub
    neg_norm = [0xbf80_0000, 0xc248_0000]            # -1.0, -50.0

    out = []
    for x in nans:
        for op in OPS:
            out.append((op, x, ANY_NAN))

    for x in zeroish_p + zeroish_n:
        out.append(('exp2', x, ONE))                 # exp2(+-0) = 1
        out.append(('log2', x, NINF))                # log2(+-0) = -inf
    for x in zeroish_p:
        out.append(('rcp',   x, PINF))
        out.append(('rsqrt', x, PINF))
    for x in zeroish_n:
        out.append(('rcp',   x, NINF))
        out.append(('rsqrt', x, NINF))               # flushed before the sign test

    out += [('exp2', PINF, PINF), ('exp2', NINF, PZERO),
            ('log2', PINF, PINF), ('log2', NINF, ANY_NAN),
            ('rcp',  PINF, PZERO), ('rcp',  NINF, NZERO),
            ('rsqrt', PINF, PZERO), ('rsqrt', NINF, ANY_NAN)]

    for x in neg_norm:
        out.append(('log2',  x, ANY_NAN))
        out.append(('rsqrt', x, ANY_NAN))
    return out


def f2b(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def b2f(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xffffffff))[0]


def exact(op, xd):
    ln2 = Decimal(2).ln()
    if op == 'exp2':
        return (xd * ln2).exp()
    if op == 'log2':
        return xd.ln() / ln2
    if op == 'rcp':
        return Decimal(1) / xd
    return Decimal(1) / xd.sqrt()


def bounds(op, bits):
    """(lo, hi) as binary32 bit patterns, or a marker for a special case."""
    x = b2f(bits)
    e = (bits >> 23) & 0xff
    f = bits & 0x7fffff
    sign = bits >> 31

    if e == 0xff:
        return ('special', None)
    if e == 0:                      # sec 8.8 flushes subnormal inputs
        return ('special', None)

    xd = Decimal(x)
    if op in ('log2', 'rsqrt') and sign:
        return ('special', None)

    # exp2 of a large argument overflows the decimal context long before it
    # overflows binary32's exponent; both are out-of-range for an accuracy
    # check, so they are filtered here rather than raised.
    try:
        v = exact(op, xd)
    except Exception:
        return ('special', None)
    if v == 0:
        return ('special', None)
    lo = v * (Decimal(1) - TOL)
    hi = v * (Decimal(1) + TOL)
    if v < 0:
        lo, hi = hi, lo
    # log2 in [0.5, 2] is an absolute bound, per sec 8.8.
    if op == 'log2' and Decimal('0.5') <= xd <= Decimal(2):
        lo, hi = v - TOL, v + TOL

    # Out-of-range results are specified rather than approximated, so they
    # are checked exactly instead of being skipped. sec 8.8 flushes a
    # subnormal result to zero, and an overflow saturates to infinity. The
    # comparison is in Decimal: a Python double underflows to 0.0 well above
    # binary32's smallest subnormal and would turn a specified flush into a
    # nonsense bound, which is how this was found.
    F32_MAX = Decimal('3.4028234663852886e38')
    F32_MIN_NORMAL = Decimal('1.1754943508222875e-38')
    mag = abs(v)
    if mag < F32_MIN_NORMAL:
        z = 0x8000_0000 if v < 0 else 0
        return (z, z)                      # flushed, exactly
    if mag > F32_MAX:
        z = 0xff80_0000 if v < 0 else 0x7f80_0000
        return (z, z)                      # saturated, exactly

    lof, hif = float(lo), float(hi)
    # Widen by one ulp each way so the reference's own binary32 rounding
    # cannot make a correct result look out of range.
    return (f2b(lof) - 1, f2b(hif) + 1) if lof > 0 else (f2b(lof) + 1, f2b(hif) - 1)


def sample(rng, n):
    out = set()
    for e in range(1, 255):
        for f in (0, 1, 0x400000, 0x7fffff, 0x123456):
            out.add((e << 23) | f)
    for v in (1.0, 2.0, 0.5, 4.0, 3.0, 1.5, 1e-30, 1e30, 0.25, 16.0, 1024.0):
        out.add(f2b(v))
    while len(out) < n:
        b = rng.getrandbits(32)
        if ((b >> 23) & 0xff) not in (0, 0xff):
            out.add(b)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-n', '--count', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    xs = sample(rng, args.count)
    rows = 0
    # Column 5 is the NaN flag: the bench compares by class when it is set,
    # and lo == hi is how an exactly-specified result is written.
    with open(args.output, 'w') as fh:
        for oi, op in enumerate(OPS):
            for b in xs:
                lo, hi = bounds(op, b)
                if lo == 'special':
                    continue
                fh.write(f"{oi} {b:08x} {lo & 0xffffffff:08x} "
                         f"{hi & 0xffffffff:08x} 0\n")
                rows += 1
        nspec = 0
        for op, x, want in specials():
            oi = OPS.index(op)
            if want is ANY_NAN:
                fh.write(f"{oi} {x:08x} 00000000 00000000 1\n")
            else:
                fh.write(f"{oi} {x:08x} {want:08x} {want:08x} 0\n")
            nspec += 1
            rows += 1
    print(f"{rows} vectors -> {args.output} "
          f"({len(xs)} inputs x {len(OPS)} ops, plus {nspec} sec 8.8 specials)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
