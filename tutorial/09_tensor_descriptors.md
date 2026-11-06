# 09 — Host tensor descriptors

[`examples/09_tensor_descriptors.py`](examples/09_tensor_descriptors.py)
compiles and executes descriptor-based GEMM at O0, O1 and O2:

```sh
make r2-stage-data
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python tutorial/examples/09_tensor_descriptors.py
```

The kernel receives descriptors directly, loads 16×16 blocks, transposes
its B block, accumulates with `tl.dot`, and stores through the output
descriptor. Its K loop uses a runtime argument. The 17×16×32 problem has
two row programs, including a partial final tile.

Compile a descriptor parameter with its element type and block shape:

```python
signature = {'a': 'tensordesc<fp16[16,16]>',
             'bt': 'tensordesc<fp16[16,16]>',
             'c': 'tensordesc<fp32[16,16]>',
             'K': 'i32', 'OUTPUT_DTYPE': 'constexpr'}
```

A Triton dtype such as `tl.float32` can be a constexpr. The artifact and
cache key retain its identity without requiring Triton in the runtime.

At launch, describe the device buffers using the public host class:

```python
from tnca.runtime import TensorDescriptor

args = {
    'a': TensorDescriptor(da, (M, K), (K, 1), (16, 16), 'fp16'),
    'bt': TensorDescriptor(db, (N, K), (K, 1), (16, 16), 'fp16'),
    'c': TensorDescriptor(dc, (M, N), (N, 1), (16, 16), 'fp32'),
    'K': K,
}
event = session.submit(module, args, grid=(2, 1, 1))
session.wait(event)
```

Shape and strides are measured in **elements**. The first argument is a
Session buffer, or `(buffer, byte_offset)` for a view. The dtype and block
shape must match the compiled descriptor parameter. Supported host storage
types are `fp16`, `bf16`, `fp32`, `i32`, `i8`, `e4m3` and `e5m2`; that list
describes descriptor storage, not every arithmetic combination.

Session checks that the descriptor fits the backing buffer, snapshots its
TensorMap in the launch argument storage, and holds a lease on the backing
buffer until completion. The kernel reads descriptor extents and strides
at runtime. No host pointer, fixed device address, or compiler-side test
harness is involved in the launch.

The script compares the complete output with an independent Python matrix
product and checks guard bytes past the logical tensor. It exits nonzero
on a compilation, launch, or numerical failure.

## Beyond this example

Descriptors follow the contract Triton exposes for TMA on NVIDIA GPUs:

- **Rank 1 to 5.**
- **Signed coordinates.** They may be negative, past the end, or straddle an edge.
- **Padding.** Out-of-bounds elements of a load read as zero, or as the
  dtype's NaN with `padding_option="nan"`.
- **Stores.** They write only the in-bounds bytes of their box.
- **Atomic adds and maxes** through a descriptor.
- **Row gather and scatter.**

[`tools/r38/tdesc_tests.py`](../tools/r38/tdesc_tests.py) (`make r38-tdesc`) runs one kernel per feature and
compares the outputs bit for bit with a model computed from the definition.
On the current RTL:

```text
PASS atomic_add             add bit-exact; max bit-exact
PASS gather_scatter         bit-exact
PASS host_descriptor        bit-exact
PASS rank1_copy             bit-exact; bit-exact; bit-exact
PASS rank1_relabel_in_loop  max abs error 0
PASS rank1_store_relabel    max abs error 0
PASS rank2_nan_pad          bit-exact
PASS rank3_dma              bit-exact
PASS rank3_dot              max rel err 0.00e+00
PASS rank3_small_trans      bit-exact
PASS rank4_rows             bit-exact
PASS rank5_fill             bit-exact
tensor descriptors [core-sim]: 12/12 pass
```

The full output also counts each case's instructions, and they show that
not every case uses the transfer engine:

- `rank3_dma` loads and stores with `t.ld.async`/`t.st.async`;
- `rank3_small_trans`, whose box fits one register, is gathered with
  `v.ld.gather`;
- `rank4_rows` is loaded row by row.

From PyTorch, Inductor's descriptor code generation reaches the same path
([10](10_torch_compile.md)).

Back to the [index](README.md).
