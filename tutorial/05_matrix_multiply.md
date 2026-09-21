# 05 · A matrix multiply on the matrix unit

**Run it:** `PYTHONPATH=python:compiler/python python tutorial/examples/05_matrix_multiply.py`

A 16×16×16 FP16 GEMM, built by the compiler's GEMM planner. The
interesting part is not the arithmetic — it is the machinery: tensor
descriptors, an asynchronous transfer into a register window, the matrix
unit, and reading the accumulator back out.

## Allocate first, compile second

```python
with Session(profile=PROFILE) as s:
    a_map  = s.allocate(256, name='descriptor for A')
    a_tile = s.allocate(M * K * 2, name='A')
    out    = s.allocate((M * N + N) * 4, name='C')

    def program(b, target):
        plan = plan_gemm(target, m=M, n=N, k=K, dtype='f16')
        build_gemm(target, plan, name='gemm',
                   a_desc_addr=a_map.addr, ..., out_addr=out.addr,
                   builder=b)
        b.exit()

    art = aot.compile_aot(program, source_kind='tnca-ir', ...)
```

This is the opposite order from the elementwise examples, and it is not
a style choice: the GEMM bakes its addresses into the program, so the
buffers have to exist before the code does.

## What it prints

```text
compiled  83 machine word(s) for a 16x16x16 FP16 GEMM
launched  SUCCEEDED, 1191 device cycle(s)
values    correct against a double-precision oracle
tail      16 sentinel slot(s) intact
row 0     [25.0, 39.0, 36.0, -5.0] ...
expected  [25.0, 39.0, 36.0, -5.0] ...
```

## The shape of what ran

The compiler writes it into the assembly's header:

```text
// GEMM 16x16x16 f16
//   MMA fragment:        16x16x16
//   k steps:             1
//   DMA commands:        2
//   MMA commands:        1
//   windows:             2 (8 vector registers each)
//   descriptors:         2
//   barriers:            2
```

Two asynchronous transfers bring A and B into two register windows of
eight registers each; one MMA consumes them; two barriers say when each
is done. `w.def w0, #56, #8` reserves v56..v63 as a window — a window is
a contiguous range of vector registers the transfer engine writes into,
which is why it cannot be scattered.

## The sentinel row

```python
out = s.allocate((M * N + N) * 4, name='C')   # one row past the tile
s.upload(out, struct.pack('<I', 0x7FC00000) * (M * N + N))
```

The output buffer is one row larger than the result, and every slot
starts as a quiet NaN. A store that wrote past the tile would leave the
values correct and the sentinel gone — so checking the values alone
would miss it. This costs one line and catches a whole class of bug.

## Where the cycles go

1191 cycles for 83 instructions. Most of that is not instruction issue:
the matrix unit waits on its operands, and the epilogue reads sixteen
accumulator rows out through the vector unit and stores each one. On
this shape roughly 80% of the stall cycles are the load/store unit.

That is why the optimisation work in [02](02_optimisation_levels.md)
buys proportionally less here than on a small elementwise kernel — there
is less issue time to remove.

---

Next: [06 · Errors and faults](06_errors_and_faults.md)
