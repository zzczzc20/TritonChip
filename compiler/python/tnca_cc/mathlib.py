"""Transcendentals and division, built out of the SFU.

ISA sec 8.8 gives four approximations -- `v.exp2.approx`, `v.rcp.approx`,
`v.rsqrt.approx`, `v.log2.approx` -- each specified to a relative error of
at most 2**-22. That is roughly four units in the last place of binary32,
and it is a published contract rather than an implementation detail, which
is what makes it something a compiler can reason about instead of measure.

It is also not enough on its own. `arith.divf` in the incoming IR means
IEEE division; returning something four ulp away from it and calling it a
divide is the kind of silent wrongness this backend refuses everywhere
else. So the reciprocal and the inverse square root are refined here by
one Newton-Raphson step, which squares the relative error to about 2**-44
and puts the result below the last bit of the format.

`exp` and `log` are NOT refined, because there is nothing to refine
against: the SFU bound is the accuracy of the underlying function and a
Newton step needs the inverse function to check against. What the
reduction here does buy is that the change of base does not ADD to that
bound -- see `exp` below. The resulting accuracy is stated per function
and is the one this compiler promises.

Every function guards its special values explicitly. A Newton step on an
infinity or a zero produces a NaN out of `inf * 0`, so the refinement has
to be thrown away in exactly the cases where the unrefined SFU result was
already exactly right -- which is what the `ordered` test at the end of
each function is doing.
"""

# The accuracy this module promises, as a relative error, for a finite
# input in the function's normal range. Quoted in the diagnostics and in
# doc/support_matrix.md; a change here is a change to a published number.
ACCURACY = {
    'div': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rcp',
    'div_ieee': 'correctly rounded (IEEE); a faithful quotient and one residual correction',
    'rcp': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rcp',
    'sqrt': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rsqrt',
    'rsqrt': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rsqrt',
    'exp': '<= 2**-22 relative, the ISA sec 8.8 exp2 bound',
    'exp2': '<= 2**-22 relative, the ISA sec 8.8 bound',
    'log': '<= 2**-22, absolute on [0.5, 2] and relative elsewhere',
    'log2': '<= 2**-22, absolute on [0.5, 2] and relative elsewhere',
    # No SFU: integer and fma arithmetic only, so these are exact functions
    # of their input on every implementation of the vector ISA. The bound
    # is the measured maximum over all 2**32 binary32 inputs against a
    # float64 reference (tools/performance_harness/transcendental_numeric.py).
    'log1p': '<= 1 ulp over every binary32 input (measured: all 2**32)',
    'sin': '<= 1 ulp over every binary32 input, full range (measured: all 2**32)',
    'cos': '<= 1 ulp over every binary32 input, full range (measured: all 2**32)',
}

LOG2E = 1.4426950408889634    # log2(e)
LN2 = 0.6931471805599453      # ln(2)


def _keep_unrefined(b, refined, unrefined):
    """`refined` where it is a number, `unrefined` where it is not.

    The Newton step multiplies the SFU result by its own argument. On an
    input where the exact answer is 0 or an infinity that product is
    `0 * inf`, so the step yields a NaN and the value to keep is the one
    the SFU already returned -- which sec 8.8's special-value table makes
    exactly right for those inputs. Testing the refined value for
    NaN-ness is therefore the same question as "was this a special
    case", and it asks it without enumerating them.
    """
    return b.vselect(b.vcmp('o', refined, refined), refined, unrefined)


def rcp(b, k, x, dtype='f32'):
    """1/x, refined."""
    r0 = b.vun('rcp', x, dtype=dtype, comment='ISA 8.8: <= 2**-22 relative')
    # e = x*r0 - 1, formed with one rounding so it is the exact residual
    # for every input where it matters.
    e = b.vfms(x, r0, k(1.0), dtype=dtype)
    # -(r0*e - r0) = r0*(2 - x*r0), the Newton step for a reciprocal.
    t = b.vfms(r0, e, r0, dtype=dtype)
    r1 = b.vun('neg', t, dtype=dtype, comment='Newton step: r0*(2 - x*r0)')
    return _keep_unrefined(b, r1, r0)


def div(b, k, a, y, dtype='f32'):
    """a/y.

    A reciprocal and a multiply, not a divide: the ISA has no divide
    (sec 8.2). The refinement is what makes the two spellings agree --
    an unrefined `a * rcp(y)` is about four ulp out, which `arith.divf`
    did not ask for.
    """
    return b.vbin('mul', a, rcp(b, k, y, dtype=dtype), dtype=dtype,
                  comment='a/y as a refined reciprocal (ISA has no divide)')


def div_ieee(b, k, a, y, dtype='f32'):
    """a/y correctly rounded (IEEE 754 division, round to nearest even).

    The faithful quotient q0 = a * rcp(y) and one correction: the residual
    t = q0*y - a is exact in one FMA, and q1 = q0 - t*rcp(y) rounds once to
    the nearest binary32 (Markstein's correction). Checked against exact
    rational division in tools/performance_harness/fp8_block_reference.py.
    Where the correction is NaN -- a zero, infinite or NaN operand, where
    q0 is already the IEEE answer -- q0 is kept.
    """
    r = rcp(b, k, y, dtype=dtype)
    q0 = b.vbin('mul', a, r, dtype=dtype, comment='a/y: faithful quotient')
    t = b.vfms(q0, y, a, dtype=dtype)
    q1 = b.vun('neg', b.vfms(t, r, q0, dtype=dtype), dtype=dtype,
               comment='a/y: one correction to the IEEE quotient')
    return _keep_unrefined(b, q1, q0)


def rsqrt(b, k, x, dtype='f32'):
    """1/sqrt(x), refined."""
    r0 = b.vun('rsqrt', x, dtype=dtype,
               comment='ISA 8.8: <= 2**-22 relative')
    rr = b.vbin('mul', r0, r0, dtype=dtype)
    # u = x*r0^2 - 3, and r1 = (-0.5*r0) * u = 0.5*r0*(3 - x*r0^2),
    # which is the Newton step for an inverse square root.
    u = b.vfms(x, rr, k(3.0), dtype=dtype)
    half = b.vbin('mul', r0, k(-0.5), dtype=dtype)
    r1 = b.vbin('mul', half, u, dtype=dtype,
                comment='Newton step: 0.5*r0*(3 - x*r0^2)')
    return _keep_unrefined(b, r1, r0)


def sqrt(b, k, x, dtype='f32'):
    """sqrt(x) as x * rsqrt(x).

    The special values do not come out of that product: `0 * inf` and
    `inf * 0` are both NaN where the answers are +0 and +inf. For all
    three of -0, +0 and +inf the answer IS the input, so the fallback is
    `x` -- except below zero, where sqrt is a NaN and `x` is not.
    """
    q = b.vbin('mul', x, rsqrt(b, k, x, dtype=dtype), dtype=dtype)
    # sqrt(-0) = -0 and sqrt(+inf) = +inf: the input, unchanged. Below
    # zero it is a qNaN, and the input is emphatically not that.
    below = b.vcmp('lt', x, k(0.0))
    fallback = b.vselect(below, k(float('nan')), x)
    return _keep_unrefined(b, q, fallback)


def exp2(b, k, x, dtype='f32'):
    """2**x, straight from the SFU."""
    return b.vun('exp2', x, dtype=dtype,
                 comment='ISA 8.8: <= 2**-22 relative')


def exp(b, k, x, dtype='f32'):
    """e**x as 2**(x*log2(e)).

    The change of base is where this could go wrong. Rounding
    `x*log2(e)` to binary32 perturbs the argument by up to |x|*2**-24,
    and exp2 amplifies an argument error by ln(2) per unit -- so at the
    top of the range, x near 88, the naive spelling loses about five
    decimal digits and lands nowhere near the 2**-22 the SFU promises.

    So the product is kept in two pieces. `t` is the rounded high part
    and `d` is its exact error, recovered with a single-rounded fms;
    2**(t+d) is 2**t * 2**d, and 2**d for a tiny d is 1 + d*ln(2) to
    well past the format's precision. One fma puts that correction back.
    """
    t = b.vbin('mul', x, k(LOG2E), dtype=dtype)
    d = b.vfms(x, k(LOG2E), t, dtype=dtype,
               comment='the exact rounding error of x*log2(e)')
    y = exp2(b, k, t, dtype=dtype)
    c = b.vbin('mul', d, k(LN2), dtype=dtype)
    # y*(1 + c) = 2**t * 2**d, with the correction applied in one fma.
    r = b.vfma(y, c, y, dtype=dtype, comment='2**t * 2**d')
    # At +-inf the correction is `inf - inf`, hence NaN, and the
    # uncorrected exp2 is already exactly +inf or +0.
    return _keep_unrefined(b, r, y)


def log2(b, k, x, dtype='f32'):
    """log2(x), straight from the SFU."""
    return b.vun('log2', x, dtype=dtype,
                 comment='ISA 8.8: <= 2**-22, absolute on [0.5, 2]')


def log(b, k, x, dtype='f32'):
    """ln(x) as log2(x)*ln(2).

    One rounding on top of the SFU's own error. There is no cheap
    refinement here: a Newton step would need `exp`, which is itself
    only accurate to 2**-22, so it would check the answer against
    something no better than the answer.

    Note what sec 8.8 promises near 1: an ABSOLUTE error of 2**-22 on
    [0.5, 2]. ln(1+e) for a small e is about e, so the relative error
    there is unbounded. That is the architecture's contract and not
    something this lowering can improve; a kernel that needs log1p
    accuracy needs a log1p.
    """
    return b.vbin('mul', log2(b, k, x, dtype=dtype), k(LN2), dtype=dtype,
                  comment='ln(x) = log2(x) * ln(2)')


def sigmoid(b, k, x, dtype='f32'):
    """1/(1 + e**-x).

    Written as a reciprocal of the sum rather than `e**x/(1+e**x)`
    because the latter overflows to `inf/inf` for a large positive x,
    where the answer is 1.
    """
    e = exp(b, k, b.vun('neg', x, dtype=dtype), dtype=dtype)
    return rcp(b, k, b.vbin('add', e, k(1.0), dtype=dtype), dtype=dtype)


# ---- functions built without the SFU ---------------------------------------
# log1p, sin and cos use only the vector integer unit and IEEE fma/add/mul,
# each a correctly rounded operation, so what they return is a fixed
# function of the input -- no "<= 2**-22" approximation enters. They are
# written against the same builder calls as everything above, which is also
# what lets tools/performance_harness/transcendental_numeric.py evaluate
# THIS code, instruction for instruction, on all 2**32 inputs.

def _ibin(b, op, x, y):
    return b.vbin(op, x, y, dtype='i32')


def _int(b, value):
    """A whole-vector 32-bit integer constant (two's complement pattern)."""
    value &= 0xFFFFFFFF
    return b.vmovi(value - (1 << 32) if value & 0x80000000 else value)


def _horner(b, z, coeffs, k):
    """coeffs[0] + z*(coeffs[1] + z*(...)), one fma per step."""
    acc = k(coeffs[-1])
    for c in coeffs[-2::-1]:
        acc = b.vfma(acc, z, k(c), dtype='f32')
    return acc


def _require_f32(name, dtype):
    if dtype != 'f32':
        from .diagnostics import Diagnostic
        raise Diagnostic(f'{name} is built for binary32; Inductor computes it in f32',
                         source='compiler/python/tnca_cc/mathlib.py')


# log1p(f) = f + f**2 * T(f) on f in [sqrt(1/2) - 1, sqrt(2) - 1]: T is a
# degree-9 near-minimax fit (Lawson) of (log1p(f) - f)/f**2 weighted for the
# relative error of the result, 2**-30.3 before rounding the coefficients
# to binary32.
_LOG1P_T = (-0.5, 0.3333331048488617, -0.2500002384185791, 0.20002266764640808,
            -0.16667790710926056, 0.1421671211719513, -0.12404774129390717,
            0.11906018853187561, -0.11694812774658203, 0.06736653298139572)
# ln 2 = LN2_HI + LN2_LO with LN2_HI on 16 bits, so k*LN2_HI is exact for
# every exponent k a binary32 can produce (|k| <= 128).
_LN2_HI = 0.693145751953125
_LN2_LO = 1.428606765330187e-06      # binary32(ln 2 - _LN2_HI)


def log1p(b, k, x, dtype='f32'):
    """ln(1 + x), accurate where 1 + x is not: it is what the function is for.

    1 + x = 2**k * (1 + f) with 1 + f in [sqrt(1/2), sqrt(2)), k read from
    the bits of u = fl(1 + x). The rounding of u is not allowed to leak in:
    f is formed from x itself, f = x*2**-k + (2**-k - 1), whose two terms
    are exact (the first by exponent arithmetic, the second for k <= 24)
    and whose sum is kept as f_hi + f_lo by a TwoSum. Then

        log1p(x) = k*ln2 + log1p(f_hi) + f_lo/(1 + f_hi),

    with log1p(f_hi) = f_hi + f_hi**2 * T(f_hi), f_lo/(1 + f_hi) taken as
    f_lo*(1 - f_hi) (f_lo is below half an ulp of f_hi; the dropped terms
    are 2**-26 of it), and the large parts added with a Fast2Sum so the
    result is rounded essentially once. For k >= 25 the 2**-k term is
    below 2**-25 of the result and is dropped. Special values are chosen
    explicitly at the end: tiny x (including -0 and subnormals) returns x,
    -1 gives -inf, below -1 NaN, +inf itself, NaN itself.
    """
    _require_f32('log1p', dtype)
    i = lambda c: _int(b, c)
    u = b.vbin('add', x, k(1.0), dtype='f32')
    # k: the exponent of u once [sqrt(1/2), sqrt(2)) is moved to [1, 2).
    kk = _ibin(b, 'sub', _ibin(b, 'lshr', _ibin(b, 'add', u, i(0x3f800000 - 0x3f3504f3)), i(23)), i(127))
    kc = _ibin(b, 'min', kk, i(25))
    a = _ibin(b, 'sub', x, _ibin(b, 'shl', kk, i(23)))          # x * 2**-k, exact
    s = _ibin(b, 'shl', _ibin(b, 'sub', i(127), kc), i(23))      # 2**-min(k, 25)
    c = b.vbin('sub', s, k(1.0), dtype='f32')                    # 2**-k - 1, exact for k <= 24
    f_hi = b.vbin('add', a, c, dtype='f32', comment='f = x*2**-k + (2**-k - 1)')
    v = b.vbin('sub', f_hi, a, dtype='f32')
    f_lo = b.vbin('add', b.vbin('sub', a, b.vbin('sub', f_hi, v, dtype='f32'), dtype='f32'),
                  b.vbin('sub', c, v, dtype='f32'), dtype='f32', comment='TwoSum: the exact error of f')
    z = b.vbin('mul', f_hi, f_hi, dtype='f32')
    t = _horner(b, f_hi, _LOG1P_T, k)
    kf = b.vcvt('s32', 'f32', kk)
    corr = b.vfma(b.vun('neg', f_hi, dtype='f32'), f_lo, f_lo, dtype='f32', comment='f_lo/(1 + f_hi)')
    corr = b.vfma(kf, k(_LN2_LO), corr, dtype='f32')
    lo = b.vfma(z, t, corr, dtype='f32')
    kl = b.vbin('mul', kf, k(_LN2_HI), dtype='f32', comment='k*ln2_hi, exact')
    hs = b.vbin('add', kl, f_hi, dtype='f32')
    tail = b.vbin('sub', f_hi, b.vbin('sub', hs, kl, dtype='f32'), dtype='f32', comment='Fast2Sum')
    out = b.vbin('add', hs, b.vbin('add', tail, lo, dtype='f32'), dtype='f32')
    ax = b.vun('abs', x, dtype='f32')
    out = b.vselect(b.vcmp('lt', ax, k(2.0 ** -25)), x, out)
    out = b.vselect(b.vcmp('eq', x, k(-1.0)), k(float('-inf')), out)
    out = b.vselect(b.vcmp('lt', x, k(-1.0)), k(float('nan')), out)
    out = b.vselect(b.vcmp('eq', x, k(float('inf'))), x, out)
    return b.vselect(b.vcmp('uo', x, x), x, out)


def _two_over_pi_words(n):
    """The first 32*n bits of the fraction of 2/pi, as n 32-bit words,
    computed (Machin's formula in integer fixed point) rather than typed."""
    prec = 32 * n + 96
    one = 1 << prec

    def atan_inv(m):
        total, term, k, m2 = 0, one // m, 1, m * m
        while term:
            total += term // k if (k // 2) % 2 == 0 else -(term // k)
            term //= m2
            k += 2
        return total
    pi = 16 * atan_inv(5) - 4 * atan_inv(239)
    frac = (2 * one * one) // pi                  # 2/pi * 2**prec, below 2**prec
    bits = frac >> (prec - 32 * n)
    return [(bits >> (32 * (n - 1 - j))) & 0xFFFFFFFF for j in range(n)]


# The words of 2/pi a reduction can need: an exponent up to 128 starts its
# 96-bit window at bit 105 of 2/pi.
_TWO_OVER_PI = _two_over_pi_words(7)
_PIO2_HI = 1.5707963705062866        # binary32(pi/2)
_PIO2_LO = -4.371138828673793e-08    # binary32(pi/2 - _PIO2_HI)
# sin(r) = r + r**3 * SP(r**2) and cos(r) = 1 - r**2/2 + r**4 * CP(r**2) on
# |r| <= pi/4: near-minimax fits, 2**-37.5 and 2**-33 relative.
_SIN_P = (-0.1666666716337204, 0.008333329111337662, -0.00019839302694890648, 2.7180260531167733e-06)
_COS_P = (0.04166664555668831, -0.0013887308305129409, 2.4432210921077058e-05)


def _reduce_pio2(b, k, ax):
    """|x| = q*pi/2 + r for every finite binary32: (q mod 4, r_hi, r_lo).

    Payne-Hanek in exact integer arithmetic. |x| = m * 2**e, m a 24-bit
    integer; m * 2**e * 2/pi mod 4 is m * Y mod 4, Y the 96 bits of 2/pi
    starting where their weight is 2: bits below that window contribute
    less than 2**-70, bits above it whole multiples of 4. The window is
    selected from seven constant words by the exponent and shifted into
    place; the product is three 32x32 multiplies (`v.mul`, `v.mulh.u`)
    modulo 2**96. Rounding to the nearest quadrant leaves a signed 64-bit
    fraction, normalised with `v.clz` and converted as two exact 24-bit
    halves, so r keeps ~48 significant bits however close |x| is to a
    multiple of pi/2. Below 0.5 the reduction is r = |x|.
    """
    i = lambda c: _int(b, c)
    E = _ibin(b, 'sub', _ibin(b, 'lshr', ax, i(23)), i(127))
    m = _ibin(b, 'or', _ibin(b, 'and', ax, i(0x7FFFFF)), i(0x800000))
    g0 = _ibin(b, 'add', _ibin(b, 'max', E, i(-1)), i(7))       # window start, bit index
    j = _ibin(b, 'lshr', g0, i(5))                               # 0..4
    sh = _ibin(b, 'and', g0, i(31))
    words = [0] + _TWO_OVER_PI
    at = [b.vcmp('eq', j, i(n)) for n in range(1, 5)]

    def word(t):
        out = i(words[t])
        for n in range(1, 5):
            out = b.vselect(at[n - 1], i(words[t + n]), out)
        return out
    z = [word(t) for t in range(4)]
    back = _ibin(b, 'sub', i(31), sh)
    y = [_ibin(b, 'or', _ibin(b, 'shl', z[t], sh),
               _ibin(b, 'lshr', _ibin(b, 'lshr', z[t + 1], i(1)), back)) for t in range(3)]
    r2 = _ibin(b, 'mul', m, y[2])
    h2 = _ibin(b, 'mulhu', m, y[2])
    r1 = _ibin(b, 'add', _ibin(b, 'mul', m, y[1]), h2)
    carry = b.vselect(b.vcmp('ltu', r1, h2), i(1), i(0))
    r0 = _ibin(b, 'add', _ibin(b, 'add', _ibin(b, 'mul', m, y[0]), _ibin(b, 'mulhu', m, y[1])), carry)
    q = _ibin(b, 'lshr', _ibin(b, 'add', r0, i(1 << 29)), i(30))  # nearest quadrant, mod 4
    hi = _ibin(b, 'or', _ibin(b, 'shl', r0, i(2)), _ibin(b, 'lshr', r1, i(30)))
    lo = _ibin(b, 'or', _ibin(b, 'shl', r1, i(2)), _ibin(b, 'lshr', r2, i(30)))
    sgn = _ibin(b, 'ashr', hi, i(31))
    hi = _ibin(b, 'xor', hi, sgn)            # magnitude, one's complement: off by <= 2**-64
    lo = _ibin(b, 'xor', lo, sgn)
    nz = b.vcmp('ne', hi, i(0))
    w0 = b.vselect(nz, hi, lo)
    w1 = b.vselect(nz, lo, i(0))
    lz = b.vun('clz', w0, dtype='i32')
    t0 = _ibin(b, 'or', _ibin(b, 'shl', w0, lz),
               _ibin(b, 'lshr', _ibin(b, 'lshr', w1, i(1)), _ibin(b, 'sub', i(31), lz)))
    t1 = _ibin(b, 'shl', w1, lz)
    scale = _ibin(b, 'add', lz, b.vselect(nz, i(0), i(32)))
    top = b.vcvt('u32', 'f32', _ibin(b, 'lshr', t0, i(8)))
    rest = b.vcvt('u32', 'f32', _ibin(b, 'or', _ibin(b, 'shl', _ibin(b, 'and', t0, i(0xFF)), i(16)),
                                    _ibin(b, 'lshr', t1, i(16))))
    f_hi = b.vbin('mul', top, _ibin(b, 'shl', _ibin(b, 'sub', i(103), scale), i(23)), dtype='f32')
    f_lo = b.vbin('mul', rest, _ibin(b, 'shl', _ibin(b, 'sub', i(79), scale), i(23)), dtype='f32')
    none = b.vcmp('eq', w0, i(0))
    sign = _ibin(b, 'and', sgn, i(0x80000000))
    f_hi = b.vselect(none, k(0.0), _ibin(b, 'xor', f_hi, sign))
    f_lo = b.vselect(none, k(0.0), _ibin(b, 'xor', f_lo, sign))
    r_hi = b.vbin('mul', f_hi, k(_PIO2_HI), dtype='f32')
    r_lo = b.vfms(f_hi, k(_PIO2_HI), r_hi, dtype='f32', comment='exact error of f_hi*pi/2')
    r_lo = b.vfma(f_lo, k(_PIO2_HI), b.vfma(f_hi, k(_PIO2_LO), r_lo, dtype='f32'), dtype='f32')
    small = b.vcmp('lt', E, i(-1))                               # |x| < 0.5
    return (b.vselect(small, i(0), q), b.vselect(small, ax, r_hi),
            b.vselect(small, k(0.0), r_lo), E)


def _sin_cos_reduced(b, k, r_hi, r_lo):
    """sin and cos of r_hi + r_lo, |r_hi| <= pi/4, |r_lo| <~ 2**-24 |r_hi|."""
    z = b.vbin('mul', r_hi, r_hi, dtype='f32')
    z_lo = b.vfms(r_hi, r_hi, z, dtype='f32', comment='exact error of r*r')
    hz = b.vbin('mul', z, k(0.5), dtype='f32')
    r3 = b.vbin('mul', r_hi, z, dtype='f32')
    # sin(r_hi + r_lo) = sin(r_hi) + r_lo*cos(r_hi), cos ~ 1 - z/2.
    tail = b.vfma(b.vun('neg', hz, dtype='f32'), r_lo, r_lo, dtype='f32')
    s = b.vbin('add', r_hi, b.vfma(r3, _horner(b, z, _SIN_P, k), tail, dtype='f32'), dtype='f32')
    # cos(r_hi + r_lo) = w + (((1 - w) - z/2) - z_lo/2 + z**2*CP(z) - r_hi*r_lo),
    # w = 1 - z/2 rounded: its rounding and the rounding of z are put back.
    w = b.vbin('sub', k(1.0), hz, dtype='f32')
    back = b.vbin('sub', b.vbin('sub', k(1.0), w, dtype='f32'), hz, dtype='f32')
    back = b.vfma(k(-0.5), z_lo, back, dtype='f32')
    back = b.vfma(b.vun('neg', r_hi, dtype='f32'), r_lo, back, dtype='f32')
    z2 = b.vbin('mul', z, z, dtype='f32')
    c = b.vbin('add', w, b.vfma(z2, _horner(b, z, _COS_P, k), back, dtype='f32'), dtype='f32')
    return s, c


def sincos(b, k, x, dtype='f32'):
    """(sin(x), cos(x)) for every binary32 x: one exact Payne-Hanek
    reduction, both polynomials of the reduced argument (each with its
    low part folded in), then each function's quadrant and sign.

    The frontend shares one call between math.sin and math.cos of the
    same value (a rotary table takes both); either result alone is the
    same instruction sequence for that result.
    """
    _require_f32('sin/cos', dtype)
    i = lambda c: _int(b, c)
    ax = b.vun('abs', x, dtype='f32')
    q, r_hi, r_lo, E = _reduce_pio2(b, k, ax)
    s, c = _sin_cos_reduced(b, k, r_hi, r_lo)
    odd = b.vcmp('ne', _ibin(b, 'and', q, i(1)), i(0))
    nan = b.vcmp('eq', E, i(128))                                  # +-inf, NaN
    # sin(q*pi/2 + r): sin r, cos r, -sin r, -cos r; odd in x.
    flip = _ibin(b, 'xor', _ibin(b, 'shl', _ibin(b, 'and', q, i(2)), i(30)),
                 _ibin(b, 'and', x, i(0x80000000)))
    out_s = b.vselect(nan, k(float('nan')), _ibin(b, 'xor', b.vselect(odd, c, s), flip))
    # cos(q*pi/2 + r): cos r, -sin r, -cos r, sin r; even in x.
    flip = _ibin(b, 'shl', _ibin(b, 'and', _ibin(b, 'add', q, i(1)), i(2)), i(30))
    out_c = b.vselect(nan, k(float('nan')), _ibin(b, 'xor', b.vselect(odd, s, c), flip))
    return out_s, out_c


def sin(b, k, x, dtype='f32'):
    """sin(x) for every binary32 x; see sincos."""
    return sincos(b, k, x, dtype)[0]


def cos(b, k, x, dtype='f32'):
    """cos(x) for every binary32 x; see sincos."""
    return sincos(b, k, x, dtype)[1]
