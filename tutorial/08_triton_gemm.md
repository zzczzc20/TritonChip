# 08 — A GEMM written as a Triton kernel

[`examples/08_triton_gemm.py`](examples/08_triton_gemm.py)

[05](05_matrix_multiply.md) multiplied matrices through the compiler's
GEMM planner, which drives the matrix unit and emits a tuned prologue,
MMA and epilogue. You cannot write that kernel; you ask for it.

This page is the other kind: a `@triton.jit` kernel you write yourself,
lowered to the vector unit. It is slower. It is also the one you can
change.

```sh
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python tutorial/examples/08_triton_gemm.py
```

## The kernel

```python
@triton.jit
def gemm_row_per_program(a_ptr, b_ptr, c_ptr,
                         K: tl.constexpr, N: tl.constexpr,
                         BLOCK: tl.constexpr):
    m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK)

    acc = tl.load(a_ptr + m * K + 0) * tl.load(b_ptr + 0 * N + offs_n)
    for k in tl.static_range(1, K):
        acc = acc + tl.load(a_ptr + m * K + k) * tl.load(b_ptr + k * N + offs_n)

    tl.store(c_ptr + m * N + offs_n, acc)
```

One program computes one row of `C`:

```
C[m, :] = sum over k of A[m, k] * B[k, :]
```

Lane *n* holds column *n*. So a row of `B` is exactly one vector
register, `A[m, k]` is a single scalar broadcast across it, and the
whole inner product is sixteen fused multiply-adds with no shuffles and
no reduction. The grid is the `M` dimension:

```python
ev = s.submit(mod, {'a_ptr': da, 'b_ptr': db, 'c_ptr': dc}, grid=(M, 1, 1))
```

## What it does

```
compiled  135 machine word(s) for a 16x16x16 FP32 GEMM, written in Triton
launched  SUCCEEDED, grid of 16 program(s), 10832 device cycle(s)
values    correct against a double-precision oracle
C[0,:4]   [-1.0, 9.0, -1.0, 4.0]
expected  [-1.0, 9.0, -1.0, 4.0]
```

Compare with [05](05_matrix_multiply.md): 1190 cycles for the same
16×16×16 shape on the matrix unit, against 10832 here for sixteen
programs on the vector unit — about 9× — and that is with 05 in FP16
against FP32 here. That gap is the matrix unit, and it is the reason the
planner exists. Use `tl` for the kernels you are inventing; use the
planner for the one shape it was built for.

## `tl.static_range`, not `range`

This is the single thing most likely to trip you up.

`for k in range(1, K)` with `K` a `tl.constexpr` is **not** unrolled.
Triton emits an `scf.for`, whose loop bounds arrive as `arith.bitcast`,
and neither has a TNCA lowering:

```
  a loop with a runtime bound  [UNSUPPORTED]
                               [UNSUPPORTED] no TNCA lowering for TTIR operation 'arith.bitcast'
```

`tl.static_range` unrolls while Triton traces, so what reaches the
backend is straight-line code with no control flow in it at all. That is
why the kernel above compiles and the same kernel with `range` does not.

If you want to see the unrolled result, [04](04_reading_the_compile.md)'s
`dump_dir` gives you the TTIR — 135 machine words for K=16 is what a
fully unrolled sixteen-step inner product costs.

## Where support ends, exactly

```
where this frontend's support currently ends
(these are the real errors, not hypotheticals):

  tl.dot                       [COMPILE_FAILED]
                               input_precision must be one of (). Got ieee
  a loop with a runtime bound  [UNSUPPORTED]
                               [UNSUPPORTED] no TNCA lowering for TTIR operation 'arith.bitcast'
```

`tl.dot` is refused by **Triton's own frontend**, before anything of
ours runs: the target publishes no dot precisions at all, so no argument
to `input_precision` is valid. Reaching the matrix unit from `tl.dot` is
future work; today the route to it is the planner of
[05](05_matrix_multiply.md).

Two more current limits, both from [06](06_errors_and_faults.md) and
[01](01_first_kernel.md):

- `BLOCK` must equal the vector length (16). A wider block needs a
  tensor spread over several vector registers with its index mapping
  intact, and is refused rather than half-done.
- `tl.zeros` lowers to a `tt.call`, which has no TNCA lowering — hence
  the accumulator above is initialised with the k=0 product rather than
  with zeros.

## Three bugs this example found

Worth saying plainly, because it is the argument for writing tutorials
against real hardware rather than describing them.

Writing this page surfaced three genuine compiler defects, all fixed in
the same commit:

1. **`_address` walked only the last link of the pointer chain.**
   `tl.load(a_ptr + m * K + offs)` dropped `m * K` entirely. Row 0 was
   correct and every other row returned row 0's data, with no error
   anywhere — the worst possible failure mode.
2. **A constant offset in the chain was mistaken for a per-lane one.**
   `b_ptr + k * N + offs_n` with a constexpr `k` was refused with "a
   per-lane offset combined with another offset". Constants are
   materialised lazily, so one still pending was not in the value table
   when the chain walk looked.
3. **A scalar load emitted a vector load.** `tl.load(a_ptr + m * K + k)`
   through a plain (non-tensor) pointer read sixteen elements into a
   vector register, and the following `tt.splat` then broadcast a vector
   register as though it were a scalar one. The answer was all zeros.
   `v.splat` now refuses a non-scalar operand outright, so this cannot
   come back silently.

---

Back to the [index](README.md).
