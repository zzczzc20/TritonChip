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
    'rcp': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rcp',
    'sqrt': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rsqrt',
    'rsqrt': '<= 1 ulp (faithful); one Newton step over ISA sec 8.8 rsqrt',
    'exp': '<= 2**-22 relative, the ISA sec 8.8 exp2 bound',
    'exp2': '<= 2**-22 relative, the ISA sec 8.8 bound',
    'log': '<= 2**-22, absolute on [0.5, 2] and relative elsewhere',
    'log2': '<= 2**-22, absolute on [0.5, 2] and relative elsewhere',
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
