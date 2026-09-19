"""
Exact reference for the five TNCA float formats.

The project's rule for floating point is that the reference is exact and the
hardware is measured against it -- tools/model/fp32.py does this for binary32
with Fractions rather than host floats, because host arithmetic rounds twice
and would hide exactly the bugs worth finding. The same rule applies here.

Formats, as (exponent bits, fraction bits, has infinity):

    f32   8, 23, yes   IEEE-754 binary32
    bf16  8,  7, yes   binary32's range, 16 fewer fraction bits
    f16   5, 10, yes   IEEE-754 binary16
    e5m2  5,  2, yes   OCP FP8 E5M2
    e4m3  4,  3, NO    OCP FP8 E4M3

E4M3 is the one that does not follow the IEEE pattern. Its all-ones exponent
is mostly ordinary numbers -- only S.1111.111 is NaN -- so its largest finite
value is 448, not the 240 an IEEE reading of the same bits would give, and it
has no infinity at all. Overflow saturates, which the OCP spec permits and
this implementation chooses.
"""
from fractions import Fraction

# name -> (exp bits, fraction bits, has_inf)
FORMATS = {
    'f32':  (8, 23, True),
    'bf16': (8, 7, True),
    'f16':  (5, 10, True),
    'e5m2': (5, 2, True),
    'e4m3': (4, 3, False),
}

# The selector the datapath carries, matching xls/tnca_formats.x.
CODE = {'f32': 0, 'bf16': 1, 'f16': 2, 'e5m2': 3, 'e4m3': 4}

NAN = 'nan'


def width(fmt):
    e, f, _ = FORMATS[fmt]
    return 1 + e + f


def bias(fmt):
    e, _, _ = FORMATS[fmt]
    return (1 << (e - 1)) - 1


def max_finite(fmt):
    """The largest finite magnitude, as an exact Fraction."""
    e, f, has_inf = FORMATS[fmt]
    top_exp = (1 << e) - (2 if has_inf else 1)
    top_frac = (1 << f) - (1 if has_inf else 2)
    return Fraction((1 << f) + top_frac, 1 << f) * Fraction(2) ** (top_exp - bias(fmt))


def decode(fmt, bits):
    """Bit pattern -> exact value.

    Returns a Fraction, or the string 'nan', or ('inf', sign).
    """
    e_sz, f_sz, has_inf = FORMATS[fmt]
    bits &= (1 << width(fmt)) - 1
    sign = (bits >> (e_sz + f_sz)) & 1
    exp = (bits >> f_sz) & ((1 << e_sz) - 1)
    frac = bits & ((1 << f_sz) - 1)
    sgn = -1 if sign else 1

    if exp == (1 << e_sz) - 1:
        if has_inf:
            return NAN if frac else ('inf', sign)
        # E4M3: only the all-ones fraction is NaN; the rest are numbers.
        if frac == (1 << f_sz) - 1:
            return NAN
    if exp == 0:
        if frac == 0:
            return Fraction(0) if not sign else Fraction(0)
        return sgn * Fraction(frac, 1 << f_sz) * Fraction(2) ** (1 - bias(fmt))
    return (sgn * Fraction((1 << f_sz) + frac, 1 << f_sz)
            * Fraction(2) ** (exp - bias(fmt)))


def _round_to_nearest_even(num, den):
    """Round the exact ratio num/den to an integer, ties to even."""
    q, r = divmod(num, den)
    twice = 2 * r
    if twice > den or (twice == den and (q & 1)):
        q += 1
    return q


def encode(fmt, value, sign=0):
    """Exact value -> bit pattern, round to nearest even.

    `value` is a non-negative Fraction, 'nan', or ('inf', s). `sign` carries
    the sign for the Fraction case, so that a signed zero survives.
    """
    e_sz, f_sz, has_inf = FORMATS[fmt]
    top = (1 << e_sz) - 1
    sbit = sign << (e_sz + f_sz)

    if value == NAN:
        # A canonical quiet NaN. E4M3's only NaN is the all-ones fraction.
        return (top << f_sz) | ((1 << f_sz) - 1) if not has_inf \
            else (top << f_sz) | (1 << (f_sz - 1))
    if isinstance(value, tuple) and value[0] == 'inf':
        s = value[1] << (e_sz + f_sz)
        if has_inf:
            return s | (top << f_sz)
        return s | saturate_bits(fmt)          # E4M3 has no infinity

    assert value >= 0, 'pass the magnitude and the sign separately'
    if value == 0:
        return sbit

    # Smallest subnormal step and the smallest normal.
    min_normal_exp = 1 - bias(fmt)
    step = Fraction(2) ** (min_normal_exp - f_sz)

    # Find the binade directly. Walking up from the smallest normal is
    # correct but costs a Fraction comparison per exponent, which for
    # binary32 sources is a couple of hundred per call; the bit lengths give
    # the same answer in one step.
    exp = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exp:
        exp -= 1
    elif value >= Fraction(2) ** (exp + 1):
        exp += 1
    subnormal = value < Fraction(2) ** min_normal_exp
    if subnormal:
        exp = min_normal_exp

    if subnormal:
        q = _round_to_nearest_even(value.numerator * step.denominator,
                                   value.denominator * step.numerator)
        if q >= (1 << f_sz):                    # rounded up into the normals
            return sbit | (1 << f_sz)
        return sbit | q

    scale = Fraction(2) ** (exp - f_sz)
    q = _round_to_nearest_even(value.numerator * scale.denominator,
                               value.denominator * scale.numerator)
    if q >= (2 << f_sz):                        # carried into the next binade
        q >>= 1
        exp += 1

    max_exp = (top - (1 if has_inf else 0)) - bias(fmt)
    if exp > max_exp or (exp == max_exp and q - (1 << f_sz) > _max_frac(fmt)):
        if has_inf:
            return sbit | (top << f_sz)
        return sbit | saturate_bits(fmt)

    return sbit | (((exp + bias(fmt)) & top) << f_sz) | (q - (1 << f_sz))


def _max_frac(fmt):
    _, f_sz, has_inf = FORMATS[fmt]
    return (1 << f_sz) - (1 if has_inf else 2)


def saturate_bits(fmt):
    """The largest finite magnitude's bit pattern, unsigned."""
    e_sz, f_sz, has_inf = FORMATS[fmt]
    top_exp = (1 << e_sz) - (2 if has_inf else 1)
    return (top_exp << f_sz) | _max_frac(fmt)


def convert(src, dst, bits):
    """The reference for the hardware's convert(src, dst, x)."""
    v = decode(src, bits)
    if v == NAN:
        return encode(dst, NAN)
    if isinstance(v, tuple):
        return encode(dst, v)
    sign = 1 if (v < 0 or (v == 0 and _sign_bit(src, bits))) else 0
    return encode(dst, abs(v), sign)


def _sign_bit(fmt, bits):
    e_sz, f_sz, _ = FORMATS[fmt]
    return (bits >> (e_sz + f_sz)) & 1
