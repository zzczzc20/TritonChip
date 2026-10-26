"""Triton kernels used by the end-to-end tests. A real .py file, because
@triton.jit needs its source."""
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, a + b, mask=mask)


@triton.jit
def mul_add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, a * b + a, mask=mask)


@triton.jit(noinline=True)
def scale_helper(v, s):
    """A @triton.jit helper the inliner must leave alone.

    The docstring here used to claim `noinline=True` while the
    decorator said nothing of the kind. It survived anyway, because
    nothing ran an inliner -- and when R3.7 started running one at
    every optimisation level (`tt.call` blocks seven TileBench
    operators), the second `tt.func` vanished and IR-04 and IR-05 both
    lost the thing they were testing.

    Its arguments are scalars because Triton refuses to leave a
    `noinline` function uninlined when it is called with a tensor.
    """
    return v * s


@triton.jit
def calls_a_helper(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """A kernel whose module really has two `tt.func`s.

    IR-04 needs a second function that must not be lowered into this
    one; IR-05 needs the `tt.call` to it, which has no lowering.
    """
    pid = tl.program_id(0)
    limit = scale_helper(n, 1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < limit
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, a * b, mask=mask)


@triton.jit
def special_constants_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """`other=` values the frontend must keep bit-exact. IR-06: MLIR prints
    an infinity as a raw hex bit pattern, which `float()` cannot read."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=float('inf'))
    tl.store(o_ptr + offs, a, mask=mask)


@triton.jit
def negative_zero_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=-0.0)
    tl.store(o_ptr + offs, a, mask=mask)


@triton.jit
def axpy_kernel(x_ptr, y_ptr, o_ptr, n, alpha, BLOCK: tl.constexpr):
    """y = alpha*x + y, with alpha a DYNAMIC f32 parameter.

    R2-03 CMP-01 asks for this one specifically, and the reason is `alpha`:
    an f32 scalar that arrives through the argument block rather than as a
    constant is what proves one artifact runs with different numbers.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, a * alpha + b, mask=mask)


@triton.jit
def descriptor_copy_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Both coordinates share zero; wide chunks must shift only columns."""
    src = tl.make_tensor_descriptor(x_ptr, shape=[2, BLOCK], strides=[BLOCK, 1],
                                    block_shape=[2, BLOCK])
    dst = tl.make_tensor_descriptor(o_ptr, shape=[2, BLOCK], strides=[BLOCK, 1],
                                    block_shape=[2, BLOCK])
    zero = tl.full((), 0, tl.int32)
    value = src.load([zero, zero])
    dst.store([zero, zero], value)
