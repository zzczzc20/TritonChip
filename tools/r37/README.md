# R3.7 original TileBench compatibility

`make r37` validates original TileBench JIT bodies through the public
`tnca.aot.compile_aot` and `tnca.runtime.Session` APIs at **O0, O1 and O2**.
The numerical runner loads `.tnca` artifacts, allocates and uploads buffers,
submits grids, waits for completion, downloads outputs and checks them.
It does not invoke the compiler importer or code generator directly.

```sh
make r37 TILEBENCH_SOURCE=/path/to/Tilebench TRITON_PYTHON=/path/to/triton/python
make r37-run R37_OPT_LEVELS=2
/path/to/triton/python tools/r37/aot_run.py --only matmul_fp32_fp16_fp8 --levels 0 1 2
/path/to/triton/python tools/r37/aot_tests.py
```

`r37-coverage` compiles every numerical configuration through AOT and writes
`sim/build/r37_aot_compile.json`. `r37-run` writes
`sim/build/r37_aot_run.json`, retaining per-case optimization level, original
source hash, artifact path, scratch requirement and numerical result.
The default matrix contains 149 cases/stages per level, 447 total.
Compilation alone is not a numerical pass. CI runs each level independently.

Set `TILEBENCH_SOURCE` to a local checkout; the default is `../Tilebench`.
Both `benchmarks/operators` and `tilebench/benchmarks/operators` layouts
are supported. The reference commit is
`6e2cf155dfe01f4660540d187c81b09eb55f394f`.

The source loader extracts unchanged JIT functions and literal constants,
retaining filenames and line numbers. It binds launch metadata instead of
running CUDA/PyTorch benchmark wrappers. Kernel bodies are not rewritten.
Inputs, launch sizes and independent references live in `upstream_cases.py`.
The public runtime takes `TensorDescriptor` objects for descriptor arguments
and automatically supplies the scratch storage requested by the compiler.

Tests use small RTL configurations: vector tails, multi-iteration reductions,
16×16 matrix blocks, FP16/FP32/FP8 products, packed INT8, convolution boundaries,
batched GEMM, stream-K and complete multi-stage normalization and sorting.
Radix sort executes all 16 two-bit passes. Stages consume preceding device
outputs, including the contiguous transposes in the original host wrapper;
host reference results never supply stage inputs. Inputs, unwritten output
regions and guard bytes are checked.

Session launches complete grids. Batched GEMM validates both batches;
stream-K cases launch the preceding full tiles as well as the selected
program, including the early-return program. Thus the AOT test is not a
static-address replay of a single program ID.

For diagnosis, the previous direct compiler/RTL paths remain available as
`make r37-direct-coverage` and `make r37-direct-run`. Simplified kernels remain
under `r37-representative-coverage` and `r37-representative-run`; their results
alone do not establish original-source or public-AOT compatibility.

[The GEMM tutorial](../../tutorial/08_triton_gemm.md) demonstrates masked
`tl.dot` with runtime dimensions and artifact reuse. [The descriptor
tutorial](../../tutorial/09_tensor_descriptors.md) demonstrates host-created
descriptors. Both execute through the public AOT and Session interfaces.

Coverage is for the explicit small configurations in the reports, not
arbitrary tile sizes, every upstream option or GPU performance parity.
