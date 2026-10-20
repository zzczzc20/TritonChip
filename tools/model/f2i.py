"""
f2i.py -- ISA sec 8.4 binary32 -> int32 / uint32 reference, from the ISA text.

vf.cvt.f32.s32 / vf.cvt.f32.u32: round by mod[2:0] (0 toward zero, 1 RNE,
2 toward -inf, 3 toward +inf), then saturate out-of-range results to the
integer extremes; NaN maps to 0. Computed on the exact value (Fractions),
independently of the RTL's shifter.
"""
from fp32 import classify, to_fraction

MODES = (0, 1, 2, 3)
MASK = 0xffffffff


def f32_to_int(raw, signed, mode):
    """Bit pattern of the 32-bit result."""
    kind, sign, _ = classify(raw)
    low, high = (-(1 << 31), (1 << 31) - 1) if signed else (0, (1 << 32) - 1)
    if kind == 'nan':
        return 0
    if kind == 'inf':
        return (low if sign else high) & MASK
    v = to_fraction(raw)
    q, r = divmod(v.numerator, v.denominator)        # floor division
    d = v.denominator
    if mode == 2:
        n = q
    elif mode == 3:
        n = q + int(r != 0)
    elif mode == 1:
        n = q + int(2 * r > d or (2 * r == d and q % 2 == 1))
    else:                                            # toward zero
        n = q + int(v.numerator < 0 and r != 0)
    return max(low, min(high, n)) & MASK
