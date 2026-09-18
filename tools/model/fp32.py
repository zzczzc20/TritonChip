"""
fp32.py -- exact binary32 reference semantics for the TNCA vector FP unit.

Everything here is computed with Fractions and rounded once, so it is an
independent oracle for the RTL rather than a restatement of it. Notably it does
NOT use the host's float arithmetic for fma: host `a*b+c` rounds twice.
"""

import struct
from fractions import Fraction

QNAN = 0x7fc00000

MANT_BITS = 23
EXP_BIAS = 127
EMIN = -126               # minimum normal exponent
EMAX = 127
SIG_BITS = 24             # including the implicit bit


def f2b(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def b2f(b: int) -> float:
    return struct.unpack('<f', struct.pack('<I', b & 0xffffffff))[0]


def classify(bits):
    s = (bits >> 31) & 1
    e = (bits >> 23) & 0xff
    f = bits & 0x7fffff
    if e == 0xff:
        return ('nan' if f else 'inf'), s, f
    if e == 0:
        return ('zero' if f == 0 else 'subnormal'), s, f
    return 'normal', s, f


def to_fraction(bits, ftz=False):
    """Exact value of a finite binary32, or None for inf/nan."""
    kind, s, f = classify(bits)
    if kind in ('inf', 'nan'):
        return None
    if kind == 'zero':
        return Fraction(0)
    if kind == 'subnormal':
        if ftz:
            return Fraction(0)
        v = Fraction(f, 1 << MANT_BITS) * Fraction(2) ** EMIN
    else:
        e = (bits >> 23) & 0xff
        v = Fraction((1 << MANT_BITS) | f, 1 << MANT_BITS) * Fraction(2) ** (e - EXP_BIAS)
    return -v if s else v


def round_to_f32(v: Fraction, sign_if_zero: int, ftz=False) -> int:
    """Round an exact Fraction to binary32, round-to-nearest-even."""
    if v == 0:
        return (sign_if_zero << 31)

    sign = 1 if v < 0 else 0
    a = abs(v)

    # e = floor(log2(a)), derived from the bit lengths and then corrected.
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1

    # Quantisation exponent: normal binade, or clamped to the subnormal grid.
    q_exp = max(e, EMIN) - (SIG_BITS - 1)

    n = a / (Fraction(2) ** q_exp)
    m = n.numerator // n.denominator
    rem = n - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (m & 1)):
        m += 1

    # Rounding up out of the binade lands exactly on a power of two, which is
    # representable one binade higher.
    if m == (1 << SIG_BITS):
        m >>= 1
        q_exp += 1

    exp = q_exp + (SIG_BITS - 1)

    if m == 0:
        return (sign << 31)

    if m >= (1 << (SIG_BITS - 1)):
        if exp > EMAX:
            return (sign << 31) | (0xff << 23)          # overflow -> inf
        return (sign << 31) | ((exp + EXP_BIAS) << 23) | (m & 0x7fffff)

    # subnormal
    if ftz:
        return (sign << 31)
    return (sign << 31) | m


def fma_ref(ab, bb, cb, sub_c=False, mul_only=False, ftz=False):
    """Exact reference for tnca_fp32_fma."""
    if sub_c:
        cb = cb ^ 0x80000000
    ka, sa, _ = classify(ab)
    kb, sb, _ = classify(bb)
    kc, sc, _ = classify(cb)

    if ftz:
        if ka == 'subnormal':
            ka, ab = 'zero', sa << 31
        if kb == 'subnormal':
            kb, bb = 'zero', sb << 31
        if kc == 'subnormal':
            kc, cb = 'zero', sc << 31

    if mul_only:
        kc, cb, sc = 'zero', 0, 0

    if ka == 'nan' or kb == 'nan' or kc == 'nan':
        return QNAN

    sp = sa ^ sb
    p_inf = (ka == 'inf' or kb == 'inf')
    inf_times_zero = (ka == 'inf' and kb == 'zero') or (kb == 'inf' and ka == 'zero')
    if inf_times_zero:
        return QNAN
    if p_inf and kc == 'inf' and sp != sc:
        return QNAN
    if p_inf:
        return (sp << 31) | (0xff << 23)
    if kc == 'inf':
        return (sc << 31) | (0xff << 23)

    fa = to_fraction(ab, ftz)
    fb = to_fraction(bb, ftz)
    fc = Fraction(0) if mul_only else to_fraction(cb, ftz)

    prod = fa * fb
    total = prod + fc

    if total == 0:
        # sign of an exact zero result under RNE
        if prod == 0 and fc == 0:
            zs = sp if sp == sc else 0
        elif prod == 0:
            zs = sc
        elif fc == 0:
            zs = sp
        else:
            zs = 0          # exact cancellation
        if mul_only:
            zs = sp
        return zs << 31

    return round_to_f32(total, 0, ftz)
