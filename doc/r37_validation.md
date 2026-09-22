# R3.7 original TileBench AOT validation

Validated on branch `r37-triton-coverage` on 2026-09-22.

**Each of O0, O1 and O2 passes 45/45 original operators and 149/149 numerical
cases/stages: 447/447 total, through `tnca.aot.compile_aot` and
`tnca.runtime.Session` on the `core-sim` RTL profile.**

The three levels ran as independent complete numerical runs. Every artifact's
recorded optimization level was checked; O1/O2 artifacts record the upstream
optimization pipeline. Every recorded original source hash was verified again
before submission.

TileBench source commit: `6e2cf155dfe01f4660540d187c81b09eb55f394f`.
Operator modules are unchanged. The loader preserves JIT bodies and binds
small launch configurations without executing CUDA/PyTorch host wrappers.

The numerical gate loads actual `.tnca` artifacts, uploads inputs, submits
public Session grids, waits for completion and downloads results. External
descriptors and compiler scratch travel through the runtime argument ABI;
the gate does not call the importer or code generator directly. Multi-stage
operators consume preceding device outputs, including all 16 two-bit radix
sort passes. Batched GEMM validates both batches, and stream-K exercises the
full-tile and early-return programs through complete grids.

Validation:

- AOT compilation: 447/447 numerical configurations across O0/O1/O2.
- AOT + Session numerical checks: 149/149 at each level; 45/45 operators at each level.
- Compiler: 69 unit tests, 6 IR cases, and ISA/schema consistency checks pass.
- Runtime/ABI/cache regressions: 7/7, 6/6 and 4/4 checks pass.
- New AOT contracts: 8/8 tests pass, covering descriptor bounds, backing-buffer
  leases, queued scratch isolation, allocation alignment, code/heap separation,
  typed cache keys, dtype constexpr metadata and atomic simulator publication.
- Delayed/reordered memory: 4/4 public-AOT softmax and histogram stages pass
  at O2 on `core-sim/slow`.
- Tutorials 01–07 pass. Updated tutorial 08 passes all 12 GEMM checks across
  the three levels (FP16/FP32, two runtime shapes and addresses per artifact).
  New tutorial 09 passes its three descriptor GEMMs. Both use `core-vrf-sync-v1`.
- An installed wheel, in a fresh environment with both Triton and `tnca_cc`
  absent, loads and runs the original FP16 descriptor matmul artifact: all
  256 outputs and the output guard match.

Commands: [tools/r37/README.md](../tools/r37/README.md).
Runnable examples: [tiled GEMM](../tutorial/08_triton_gemm.md) and
[host descriptors](../tutorial/09_tensor_descriptors.md).

Generated evidence is kept outside Git: `sim/build/r37_aot_compile.json`,
`r37_aot_O0.json`, `r37_aot_O1.json`, `r37_aot_O2.json`, the aggregate
`r37_aot_run.json`, and `r37_aot_slow.json`. The aggregate records all source
and artifact hashes. CI executes each level separately and archives the
numerical report and compiled artifacts.

Earlier direct-lowering reports (`r37_upstream*.json`) remain diagnostic
records; they are not the evidence for the public-AOT results above.

Correctness coverage is for these explicit small RTL configurations. It does
not establish arbitrary tile sizes, every upstream launch option, or GPU
performance parity.

| Original operator | Passing cases/stages per level |
|---|---:|
| `vector_add` | 1 |
| `mul2` | 1 |
| `relu` | 1 |
| `leaky_relu` | 1 |
| `sigmoid` | 1 |
| `fused_activation` | 1 |
| `swiglu` | 1 |
| `dropout` | 1 |
| `quantize_global` | 1 |
| `dequantize_rowwise` | 1 |
| `weight_dequant` | 1 |
| `rope` | 1 |
| `softmax` | 2 |
| `layernorm` | 2 |
| `rmsnorm` | 2 |
| `l2_norm` | 2 |
| `mean_reduction` | 1 |
| `argmax` | 2 |
| `cross_entropy` | 1 |
| `kl_divergence` | 2 |
| `batch_normalization` | 3 |
| `moe_topk_gating` | 1 |
| `histogramming` | 2 |
| `matmul_fp32_fp16_fp8` | 4 |
| `matmul_int8` | 1 |
| `batched_matmul` | 1 |
| `streamk_matmul` | 4 |
| `flash_attention` | 1 |
| `flash_decode` | 1 |
| `block_sparse_attention` | 1 |
| `linear_self_attention` | 5 |
| `1d_conv` | 1 |
| `2d_conv` | 1 |
| `3d_conv` | 1 |
| `2d_max_pooling` | 1 |
| `gaussian_blur` | 1 |
| `jacobi_stencil_2d` | 1 |
| `matrix_copy` | 1 |
| `matrix_transpose` | 1 |
| `interleave` | 1 |
| `reverse_array` | 1 |
| `destindex` | 1 |
| `bitonic_sort` | 7 |
| `radix_sort` | 80 |
| `top_k_selection` | 1 |
