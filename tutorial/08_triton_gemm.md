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
against FP32 here.

That gap is the matrix unit, and you do not have to leave Triton to
reach it: the `tl.dot` version further down this page gets there from
an ordinary kernel. The planner of [05](05_matrix_multiply.md) is
still the thing to use when you want to choose the pipeline depth and
the k-step yourself; `tl.dot` is the thing to use when you want the
product.

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

## The same product, as one `tl.dot`

Everything above is the vector unit. `tl.dot` is the matrix unit, and
the kernel for it is the one every Triton matmul tutorial has:

```python
om = tl.arange(0, M)
ok = tl.arange(0, K)
on = tl.arange(0, N)
a = tl.load(a_ptr + om[:, None] * K + ok[None, :])
b = tl.load(b_ptr + ok[:, None] * N + on[None, :])
tl.store(c_ptr + om[:, None] * N + on[None, :], tl.dot(a, b))
```

Nothing in it is written for this machine. What the backend does is
recognise that `om[:, None] * K + ok[None, :]` describes a tile:

```
compiled  164 machine word(s) for the SAME product written as tl.dot
  td.load        x2   the TensorMap snapshot
  t.ld.async     x2   the tile fetch
  m.mma.async    x1   the 16x16x16 product
  a.vread        x16  the accumulator epilogue
  scratch       512 byte(s) for the descriptors it writes
```

164 machine words against 135 for the unrolled vector version, and one
instruction doing the arithmetic instead of sixteen broadcast products
per row.

The tile arrives through the transfer engine rather than through vector
loads, and that is not a preference. A 16×16 FP16 fragment is 512 bytes
in eight registers, two rows per register (ISA sec 9C.2), while
`v.ld.h` puts one 2-byte element in each of sixteen 32-bit lanes.
Assembling the packed form in registers needs `v.pack.f16`, which sec
8.4 marks *Reserved* — so the engine is not the fast path for a tile,
it is the only one. `td.load` snapshots from memory and no instruction
builds a descriptor out of registers, which is what the 512 bytes of
scratch are for.

## Where support ends, exactly

```
where this frontend's support currently ends
(these are the real errors, not hypotheticals):

  tl.dot on f32                [UNSUPPORTED]
                               [UNSUPPORTED] a tl.dot with f32 inputs
  a loop with a runtime bound  [UNSUPPORTED]
                               [UNSUPPORTED] no TNCA lowering for TTIR operation 'scf.for'
```

`tl.dot` itself is supported; a `tl.dot` on **binary32** is not. ISA
sec 9C's `mod[3:2]` names FP16, BF16 and the two FP8 encodings and
reserves the rest, so there is no f32 matrix unit to lower it onto.
Widening the operands to FP16 behind your back would be a different
computation, so it is refused by name instead.

An integer `tl.dot` IS accepted, because there the widening is not a
choice: every INT8 is an integer under 2048, which FP16 holds exactly,
and sec 9C.1's one-rounding-per-step accumulation in binary32 is exact
for every partial sum under 2²⁴. That holds for K up to 1024, and a
larger K is refused rather than quietly approximated.

Two more current limits, both from [06](06_errors_and_faults.md) and
[01](01_first_kernel.md):

- a 1-D `BLOCK` must equal the vector length (16). A wider block needs
  a tensor spread over several vector registers with its index mapping
  intact, and is refused rather than half-done. A 2-D 16×16 block is a
  different thing and is supported — it is a tile.
- a loop whose bound is a runtime value stays an `scf.for`, which has
  no TNCA lowering. `tl.static_range` is unrolled while Triton traces.

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
