# 08 — Tiled Triton GEMM through AOT and Session

Run [`examples/08_triton_gemm.py`](examples/08_triton_gemm.py):

```sh
make r2-stage-data
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python tutorial/examples/08_triton_gemm.py
```

The example compiles an ordinary `@triton.jit` kernel containing `tl.dot`,
a runtime K loop, masked loads and masked stores. It checks FP16 and FP32
inputs independently at **O0, O1 and O2**, with FP32 output.

```python
@triton.jit
def gemm(a, b, c, M, N, K, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK)):
        kk = start * BLOCK + ks
        aa = tl.load(a + rows[:, None] * K + kk[None, :],
                     (rows[:, None] < M) & (kk[None, :] < K), other=0)
        bb = tl.load(b + kk[:, None] * N + cols[None, :],
                     (kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(aa, bb, acc, input_precision='ieee')
    tl.store(c + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))
```

`M`, `N`, and `K` are runtime arguments. `BLOCK=16` is a compile-time
constant. `range` remains a loop in the compiled program; it does not need
to be rewritten with `tl.static_range`.

Compile using the public interface:

```python
artifact = aot.compile_aot(
    gemm,
    signature={'a': '*fp16', 'b': '*fp16', 'c': '*fp32',
               'M': 'i32', 'N': 'i32', 'K': 'i32', 'BLOCK': 'constexpr'},
    constexprs={'BLOCK': 16}, profile=PROFILE,
    options={'opt_level': 2}, output='gemm.tnca')
```

Then use `Session.load`, `allocate`, `upload`, `submit`, `wait`, and
`download`, just as in [01](01_first_kernel.md):

```python
event = session.submit(module,
    {'a': da, 'b': db, 'c': dc, 'M': M, 'N': N, 'K': K},
    grid=(triton.cdiv(M, 16), triton.cdiv(N, 16), 1))
session.wait(event)
```

Each compiled artifact is reused for two launches with different dimensions
and buffer addresses: 16×16×16 and 17×19×32. The second launch exercises
both output tails, a two-dimensional grid, and two K iterations. The host
checks every result against an independent Python product, verifies both
input buffers remain intact, and checks the output guard bytes.

FP16 products can use the matrix unit. FP32 `tl.dot` uses vector FP32
multiply-add operations without narrowing the inputs to FP16. Session
allocates the scratch storage recorded in the artifact, passes its address
to the kernel, and releases it after the task completes.

These are small RTL correctness examples. Their measured cycles are printed
by the script; they are not GPU performance comparisons or a guarantee for
arbitrary tile sizes.

For host-created descriptors and descriptor bounds, continue to
[09](09_tensor_descriptors.md).
