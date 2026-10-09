"""`triton.language.extra.libdevice` for TNCA.

Kernels written against libdevice -- PyTorch Inductor's all are -- resolve its
functions through the backend's module map (compiler.TNCABackend.get_module_map).
Each function here is the Triton primitive with the same semantics, which the
TNCA lowering already implements.

Only functions whose meaning a primitive states exactly, or to within the
SFU's published bound (ISA sec 8.8), are provided. A function left out is not
approximated: a kernel that calls it fails to compile, which is the loud
failure, instead of running something close. Deliberately absent: pow (the
exp/log form is wrong for a negative base), log1p/expm1 (the obvious forms
lose the precision the functions exist for), erf/erfinv/lgamma, and the
trigonometric family.
"""
import triton
import triton.language as tl


@triton.jit
def exp(x):
    return tl.exp(x)


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def log(x):
    return tl.log(x)


@triton.jit
def log2(x):
    return tl.log2(x)


@triton.jit
def sqrt(x):
    return tl.sqrt(x)


@triton.jit
def rsqrt(x):
    return tl.rsqrt(x)


@triton.jit
def floor(x):
    return tl.floor(x)


@triton.jit
def ceil(x):
    return tl.ceil(x)


@triton.jit
def abs(x):
    return tl.abs(x)


@triton.jit
def isnan(x):
    return x != x


@triton.jit
def isinf(x):
    return tl.abs(x) == float('inf')


@triton.jit
def tanh(x):
    # sign(x) * (1 - e) / (1 + e), e = exp(-2|x|): no overflow for any x. The
    # relative error grows for |x| below about 1e-4, where 1 - e cancels;
    # the absolute error stays at the exp bound.
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -t, t)
