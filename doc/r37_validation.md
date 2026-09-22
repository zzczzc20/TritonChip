# R3.7 original TileBench validation

Validated on branch `r37-triton-coverage` on 2026-09-22.

**45/45 original operators pass numerical RTL validation: 149/149 cases and stages.**

TileBench source commit: `6e2cf155dfe01f4660540d187c81b09eb55f394f`. The operator modules are unchanged from that commit. The loader preserves JIT function bodies and binds small launch configurations; it does not run the CUDA host wrappers.

`tl.dot` cases include FP16, FP32, FP8 E4M3/E5M2, packed INT8, a two-iteration K loop, a nonzero batch index, attention, convolutions, and stream-K atomic accumulation. Multi-stage operators consume preceding RTL dumps; radix sort executes all 16 two-bit passes.

Validation:

- Original-source compilation: 45/45 operators, including every entry-point stage.
- Original-source RTL: 45/45 operators; 149/149 cases/stages in one complete run.
- Compiler checks: 69 unit tests and 6 IR cases pass; ISA/schema consistency checks pass.
- Memory behavior: original softmax, histogram and FP16 GEMM pass with reordered responses; original transpose passes with delayed memory.
- Representative regressions: 214/217 passed the initial run; the three affected stream-K cases all passed after the atomic-mask fix. The explicit combined revalidation record covers 217/217.

Commands and configuration details: [tools/r37/README.md](../tools/r37/README.md).

Generated reports: `sim/build/r37_upstream.json`, `sim/build/r37_upstream_run.json`, `sim/build/r37_memory_checks.json`, and `sim/build/r37_representative_revalidated.json`. The numerical report records source hashes, signatures, constexprs, scalar arguments, descriptor shapes, program IDs, tolerances and every stage result.

These results establish correctness for the explicit small configurations, not arbitrary tile sizes or performance parity with GPU implementations.

| Original operator | Passing cases/stages |
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
