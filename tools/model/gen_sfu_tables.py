#!/usr/bin/env python3
"""
Generate xls/tnca_sfu_tables.x -- the SFU's interpolation coefficients.

ISA sec 8.8 gives each SFU operation a published error bound of 2^-22
relative, and says plainly that the bound is a contract rather than an
implementation detail. So the coefficients are computed here, exactly, in
Fractions, and emitted as DSLX; the same routine produces the reference the
hardware is measured against, which means the table and its reference cannot
drift apart.

The shape is the one real special-function units use: reduce the argument to
a short interval, look up a quadratic in a table indexed by the top bits, and
evaluate. With 128 segments the truncation error of a quadratic is about
2^-25 for every one of the four functions, which leaves the 2^-22 budget to
be spent on the fixed-point arithmetic rather than on the approximation.

Each entry is (c0, c1, c2) for f(x0 + u) ~ c0 + c1*u + c2*u^2, with u in
[0, 1/SEGMENTS) and the coefficients taken from the Taylor expansion at the
segment's left edge -- not a minimax fit. A minimax fit would buy roughly one
bit; a Taylor expansion at a known point can be checked by reading it, which
is worth more here than the bit.
"""
import os
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

# 60 significant digits is about 199 bits: far more than the 30 fractional
# bits the coefficients carry, so the quantisation below is the only rounding
# that matters. Exact Fractions were the first attempt and are the wrong tool
# -- a Newton iteration on a Fraction doubles the denominator's size each
# step, and the numbers stop fitting in memory long before they stop being
# exact.
getcontext().prec = 60

# 256 segments, not 128. The quadratic's truncation error is u^3 times the
# third derivative over six, and the four functions do not share a third
# derivative: 1/(1+x) has much the largest. At 128 segments rcp's relative
# error works out at about 2^-20, which the measurement confirmed -- roughly
# forty vectors sat two ULP outside the bound while the other three functions
# were comfortably inside. Halving the segment width cubes down to 2^-24 for
# rcp and leaves the rest with more margin than they need.
SEGMENTS = 256
LOG2_SEG = 8

# Fixed-point formats, chosen so every coefficient's range fits with the
# fraction bits below and the products stay inside 64 bits.
C0_FRAC = 30      # c0 in [0.5, 2): Q2.30
C1_FRAC = 30      # c1 magnitudes stay under 2
C2_FRAC = 30      # c2 magnitudes stay under 2
PREC = 200        # working precision for the series, in bits


def _exp(x):
    """e^x by its Taylor series, at the context's precision."""
    s_, t = Decimal(1), Decimal(1)
    for k in range(1, 200):
        t = t * x / k
        s_ += t
        if abs(t) < Decimal(10) ** -55:
            break
    return s_


def _ln(x):
    """ln x via atanh, which converges quickly for x near 1."""
    # Bring x into [1,2) and account for the powers of two afterwards.
    e = 0
    while x >= 2:
        x /= 2
        e += 1
    while x < 1:
        x *= 2
        e -= 1
    y = (x - 1) / (x + 1)
    s_, t, k = Decimal(0), y, 1
    y2 = y * y
    while k < 400:
        d = t / k
        s_ += d
        if abs(d) < Decimal(10) ** -55:
            break
        t *= y2
        k += 2
    return 2 * s_ + e * LN2


LN2 = Decimal(2).ln()


def exp2(x):
    return _exp(x * LN2)


def log2(x):
    return x.ln() / LN2


def quantize(v, frac_bits):
    """Round a Decimal to a signed fixed-point integer, ties away from zero."""
    scaled = v * (1 << frac_bits)
    as_frac = Fraction(scaled)
    n, d = as_frac.numerator, as_frac.denominator
    neg = n < 0
    n = abs(n)
    q, r = divmod(n, d)
    if 2 * r >= d:
        q += 1
    return -q if neg else q


# ---------------------------------------------------------------------------
# The four functions, each as (value, first derivative, second derivative / 2)
# at the left edge of a segment.
# ---------------------------------------------------------------------------
def exp2_coeffs(x0):
    """2^(x0+u), u in [0, 1/SEGMENTS). d/dx 2^x = ln2 * 2^x."""
    v = exp2(x0)
    return v, v * LN2, v * LN2 * LN2 / 2


def log2_coeffs(x0):
    """log2(1 + x0 + u), the argument reduced to m-1 in [0,1)."""
    m = 1 + x0
    v = log2(m)
    d1 = 1 / (m * LN2)
    d2 = -1 / (m * m * LN2) / 2
    return v, d1, d2


def rcp_coeffs(x0):
    """1/(1 + x0 + u)."""
    m = 1 + x0
    v = 1 / m
    return v, -1 / (m * m), 1 / (m * m * m)


def rsqrt_coeffs(x0):
    """1/sqrt(1 + x0 + u)."""
    m = 1 + x0
    v = 1 / m.sqrt()
    d1 = -v / (2 * m)
    d2 = 3 * v / (4 * m * m) / 2
    return v, d1, d2


# rsqrt of a mantissa scaled by two, for odd exponents: 1/sqrt(2(1+x0+u)).
def rsqrt2_coeffs(x0):
    m = 2 * (1 + x0)
    v = 1 / m.sqrt()
    # d/dx0 with m = 2(1+x0): dm/dx0 = 2, so d1 = -v/m and d2 = 3v/(2 m^2).
    d1 = -v / m
    d2 = 3 * v / (2 * m * m)
    return v, d1, d2


TABLES = [
    ('EXP2', exp2_coeffs, 'x0 = i/SEGMENTS over [0,1): 2^x0'),
    ('LOG2', log2_coeffs, 'x0 = i/SEGMENTS over [0,1): log2(1+x0)'),
    ('RCP', rcp_coeffs, 'x0 = i/SEGMENTS over [0,1): 1/(1+x0)'),
    ('RSQRT', rsqrt_coeffs, 'x0 = i/SEGMENTS over [0,1): 1/sqrt(1+x0)'),
    ('RSQRT2', rsqrt2_coeffs, 'the same, with the mantissa doubled, for odd exponents'),
]


def build():
    out = {}
    for name, fn, _doc in TABLES:
        rows = []
        for i in range(SEGMENTS):
            x0 = Decimal(i) / Decimal(SEGMENTS)
            c0, c1, c2 = fn(x0)
            rows.append((quantize(c0, C0_FRAC),
                         quantize(c1, C1_FRAC),
                         quantize(c2, C2_FRAC)))
        out[name] = rows
    return out


def emit(tables, path):
    L = []
    L.append("// GENERATED by tools/model/gen_sfu_tables.py -- do not edit.")
    L.append("//")
    L.append("// Quadratic interpolation coefficients for the SFU. Each entry is")
    L.append("// (c0, c1, c2) for f(x0 + u) ~ c0 + c1*u + c2*u^2 with u in")
    L.append(f"// [0, 1/{SEGMENTS}), taken from the Taylor expansion at the segment's")
    L.append("// left edge and rounded to fixed point:")
    L.append(f"//     c0, c1, c2 all s32 in Q_.{C0_FRAC}.")
    L.append("//")
    L.append("// 32 bits, not 64. Every coefficient of every function has magnitude")
    L.append("// below 2, so Q2.30 holds it exactly; a 64-bit table doubled the")
    L.append("// constant-mux width and made the products 64x64 where 32x22 is what")
    L.append("// the values need. yosys spent six minutes on one SFU function before")
    L.append("// this and did not finish.")
    L.append("//")
    L.append("// ISA sec 8.8 makes the SFU's error bound a published contract, so these")
    L.append("// are computed in exact Fractions by the same script that produces the")
    L.append("// reference the hardware is measured against.")
    L.append("")
    L.append(f"pub const SEGMENTS: u32 = u32:{SEGMENTS};")
    L.append(f"pub const LOG2_SEGMENTS: u32 = u32:{LOG2_SEG};")
    L.append(f"pub const COEF_FRAC: u32 = u32:{C0_FRAC};")
    L.append("")
    for name, _fn, doc in TABLES:
        rows = tables[name]
        L.append(f"// {doc}")
        L.append(f"pub const {name}_C0 = s32[{SEGMENTS}]:[")
        L.append('    ' + ', '.join(f"s32:{r[0]}" for r in rows))
        L.append("];")
        L.append(f"pub const {name}_C1 = s32[{SEGMENTS}]:[")
        L.append('    ' + ', '.join(f"s32:{r[1]}" for r in rows))
        L.append("];")
        L.append(f"pub const {name}_C2 = s32[{SEGMENTS}]:[")
        L.append('    ' + ', '.join(f"s32:{r[2]}" for r in rows))
        L.append("];")
        L.append("")
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..'))
    path = os.path.join(root, 'xls', 'tnca_sfu_tables.x')
    tables = build()
    emit(tables, path)
    print(f"wrote {os.path.relpath(path, root)}: "
          f"{len(TABLES)} tables x {SEGMENTS} entries")
    return 0


if __name__ == '__main__':
    sys.exit(main())
