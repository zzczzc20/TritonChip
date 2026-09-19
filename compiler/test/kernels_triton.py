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
