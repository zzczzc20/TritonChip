# R3.7 original TileBench compatibility

`make r37` loads and executes the unchanged JIT function bodies from a local
TileBench checkout. Set `TILEBENCH_SOURCE=/path/to/Tilebench` and
`TRITON_PYTHON=/path/to/triton/python` if needed. Both known repository layouts
(`benchmarks/operators` and `tilebench/benchmarks/operators`) are supported.

```sh
make r37-coverage
make r37-run
# Focus a numerical regression:
/path/to/triton/python tools/r37/upstream_run.py --only flash_attention
```

The loader extracts JIT functions and literal constants, retaining original
filenames and line numbers. It does not rewrite their bodies or execute the
CUDA/PyTorch benchmark wrappers. Launch sizes, argument types, descriptors,
and inputs are specified in `upstream_cases.py`. The reference checkout used
for development is TileBench commit
`6e2cf155dfe01f4660540d187c81b09eb55f394f`.

`r37_upstream.json` reports compilation of all entry points. The separate
`r37_upstream_run.json` reports numerical execution, with source SHA-256,
launch configuration and per-stage results. Compilation alone is not a pass.
An operator passes only when all its configured cases pass; absent cases fail.

Tests deliberately use small dimensions for RTL simulation: vector tails,
37-element multi-iteration reductions, 16x16 matrix blocks, masked convolution
and spatial boundaries, a nonzero batch index, and the complete multi-stage
histogram, normalization, attention and sorting algorithms. Radix sort runs
all 16 two-bit passes with the original 16-bit packed count fields. Stages
consume decoded RTL outputs, including the contiguous transposes performed by
the original host wrapper. Host reference calculations never supply stage
inputs. Input buffers, unwritten output regions and guard bytes are checked.

The simulation memory reserves 128 KiB for emitted code, followed by input
buffers and descriptor/spill storage. Matrix shapes stay small; additional
memory accommodates compiler expansion and is not larger problem data.

The older simplified operator kernels remain as compiler regression tests:
`make r37-representative-coverage` and `make r37-representative-run`. Their
results do not establish compatibility with original TileBench sources.

Coverage is for the explicit small configurations in the report. It does not
claim arbitrary tile sizes, performance parity with CUDA, or every dtype and
shape accepted by the upstream host wrappers.
