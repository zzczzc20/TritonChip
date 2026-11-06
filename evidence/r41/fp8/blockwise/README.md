# Blockwise-scaled `torch._scaled_mm` on `tnca` through Inductor

Branch `r41-sw`. This is development evidence. The SystemC/TLM results are uncalibrated model predictions,
and the RTL result is a simulation. No board was used.

**Result: done, generically.** On tnca, BlockWise1x128 and BlockWise128x128 `_scaled_mm` (v1) and
`_scaled_mm_v2` / `F.scaled_mm` now compile through Inductor to one Triton kernel. It runs the FP8 operands on
the FP8 MMA, and its output is **bitwise equal** to a plain-torch reference in the device's order:

- **SystemC/TLM model:** 16/16 cases;
- **single-core RTL simulation:** 2/2 small cases.

There is no host fallback, and nothing in the change names a model or a layer.

## Why it failed before (torch 2.14.0)

- **Eager:** CPU eager refuses blockwise scales. `_scaled_mm_cpu` raises `ValueError: Invalid scaling
  configuration`, and `_scaled_mm_v2` raises "unsupported recipe(s)". So there is no host reference and no host
  fallback.
- **Inductor:**
  - Both `tuned_scaled_mm` and `tuned_scaled_mm_v2` (`_inductor/kernel/mm.py`) offer blockwise only through
    `scaled_mm_device_tma_main_loop_scaling`. This is a persistent template that uses `make_tensor_descriptor`.
  - `use_triton_tma_template` gates it. It needs `config.triton.enable_persistent_tma_matmul`, which is off by
    default. It also needs `can_use_tma`, which calls `has_triton_tma_device()` and so requires a CUDA device of
    capability 9 or higher.
  - The mm template is offered only for the epilogue (tensorwise/rowwise) recipes.
  - So tnca had no choice, and the call failed with `NoValidChoicesError`.
- **Enabling the upstream template for tnca's descriptors was not the route.** Three reasons:
  1. The template is CUDA-only by construction: a persistent grid over `NUM_SMS`, and the TMA gate above.
  2. It dequantizes before the dot (`a * scale_a_block`, then an FP32 `tl.dot`), so the FP8 MMA would not be used.
  3. It reads the scales as if they were contiguous, whatever their strides. The CUDA kernels require
     outer-dim-major 1x128 scales.

## What was done

`python/tnca/scaled_mm_blockwise.py` is installed by the device policy (`TNCA_POLICY_FP8_CONVERT`,
`torch_device/policy.py`). It wraps the v1 and v2 lowerings, and only a blockwise call on tnca takes the new path.
Tensorwise, rowwise and all other `_scaled_mm` calls keep their lowering.

**The template.** `TritonTemplate` `tnca_scaled_mm_blockwise` uses `mm_grid` and is the Hub block kernel's loop:

```
for kb in range(cdiv(K, 128)):           # ascending
    acc += tl.dot(a[:, kb], b[kb, :]) * a_s[:, None] * b_s[None, :]
```

- The FP8 operands go to `tl.dot` as stored, which issues `m.mma.async.k32.e4m3`.
- Scale strides are read from the graph, so any scale layout works.
- `BLOCK_M` is the GEMM tile's M (16). `BLOCK_N` and `BLOCK_K` are 128:
  - `BLOCK_K` is the scale's K block.
  - At `BLOCK_N` = 16 the compiler runs out of vector registers. At 32 and 64 the register blocks have mismatched
    widths. 128 is the tile that the compiler's extended lowering already takes for the Hub kernel.
- **E4M3 x E5M2.** Each 128-deep block is computed as 16-deep `tl.dot(a, b, part)` steps into one binary32
  partial. The compiler builds its exact FP16 substitute only for 16-deep K, and this keeps the same product-adds
  in the same order.
- **Epilogue.** The bias, an FP8 output's `1/scale_result` and the cast to the output dtype are pointwise nodes.
  They fuse into the template's store and are rounded once, as upstream's scaled-mm epilogue does.

**Scale semantics.** The scale tile is 128 along K, and 1 (1x128) or 128 (128x128) along M or N.

- **v1** uses the meta function's shapes and pairs:
  - scale_a has shape (rows, kb);
  - scale_b has shape (kb, cols);
  - the pairs are (1x128, 128x128), (1x128, 1x128) and (128x128, 1x128).
- **v2** takes the recipe it is given. Its shapes follow the CUDA kernels (ATen `ScaledBlas.cpp`):
  - 1x128 scales are (M or N, kb);
  - 128x128 scales are (round_up(kb, 4), M/128 or N/128);
  - the transposed shape is accepted where the two shapes differ;
  - padding K blocks are never read.
- **Rejected pair.** v2 128x128 x 128x128 is refused (`NotImplementedError`): torch defines no such pairing.

## Numerics

The reference is `blockwise_reference` in `tools/performance_harness/fp8_blockwise_cases.py`. It is plain torch,
in the device's order:

1. For each 128-deep K block, ascending, form a binary32 partial: one rounding per exact FP8 product-add, in
   ascending K, from +0.0 (`fp8_cases.per_k_fma`).
2. Then `* sa`, `* sb` and `+ acc`, each rounded once.
3. Then add the bias, if any.
4. Round once to the output dtype.

Agreement is required **bit for bit, with zero tolerance**. The scales are random (not powers of two), so every
multiply rounds.

## Cases (`systemc-tlm-grid`)

Settings: `TNCA_TLM_MEMORY_BYTES=0x200000000`, `TNCA_PROFILE=core-vrf-sync-v5-x8`, `TNCA_IEEE_DIV=1`,
`TNCA_DRAM_PATH=fast`.

**Result: 16 pass, 0 fail.** Every case is one fused kernel with no host fallback, and its census shows FP8 MMAs.

| API | pair | M×N×K | out | bits |
|---|---|---|---|---|
| v2 | 1x128 × 128x128 | 1×256×256, 7×200×328, 129×136×200 | bf16, fp32, bf16 | 0 mismatches |
| v2 | 1x128 × 1x128 | 1×136×328, 7×256×256, 129×200×200 | fp32, bf16, bf16 | 0 |
| v2 | 128x128 × 1x128 | 1×200×256, 7×136×328, 129×256×200 | bf16, fp32, bf16 | 0 |
| v1 | each of the three pairs | 7×200×328, 129×136×256, 129×200×328 | bf16, fp32, bf16 | 0 |
| v2 | 1x128 × 128x128 + bias | 7×256×256 | bf16 | 0 |
| v2 | 1x128 × 1x128, E4M3 × E5M2 | 7×136×200 | bf16 | 0 (`m.mma.async.k16.f16`) |
| v2 | 1x128 × 128x128, contiguous (non-CUDA) scale strides | 7×136×200 | fp32 | 0 |
| v1 | 1x128 × 1x128 + bias | 7×136×328 | bf16 | 0 |

The shapes cover M = 1, 7 and 129, and N and K values that are not multiples of 128 (136, 200, 328).

**RTL** (`sim-file`, single-core Icarus, `core-vrf-sync-v5-x8`, `--quick`): **2/2 pass, bitwise**:

- v2 1x128 × 128x128, 1×136×200 → bf16;
- v1 128x128 × 1x128, 7×16×144 → fp32.

## No regression (final tree `files:b781546f41d1…`)

The model runs were made on the binary `~/r41-sw-mb/tnca-chip`. The model itself is unchanged.

| Check | Result |
|---|---|
| `make compiler` | generated-file checks pass; `compiler-unit` **143 tests pass**; `compiler-ir` **6/6 pass** |
| `default_change_audit.py`, compared with r41s1 (`r41-fp8-audit/run-ctrl`) | **113 unchanged, 0 changed** |
| `fp8_cases.py` (full; TLM, 8 GiB window, x8, IEEE div) | **100/100 pass** |
| `fp8_kernels_cases.py` (full; same settings) | **35/35 pass** |

## Archive (`blockwise-evidence.tgz`)

| Path | Contents |
|---|---|
| `tlm/` | `report.json` and log of the full run, the generated wrapper and kernel per case, dumps (`.ttir`, `.s`) |
| `rtl/` | the same for the RTL quick run |
| `regression/` | status, `make compiler` log, audit compare JSON, the `fp8_cases` and `fp8_kernels_cases` reports |

## Reproduce (GB10)

```sh
export TNCA_BACKEND=systemc-tlm-grid TNCA_TLM_BINARY=... TNCA_TLM_MEMORY_BYTES=0x200000000 \
       TNCA_PROFILE=core-vrf-sync-v5-x8 TNCA_IEEE_DIV=1 TNCA_DRAM_PATH=fast PYTHONPATH=$T/python:$T/compiler/python
$TORCH_PYTHON $T/tools/performance_harness/fp8_blockwise_cases.py --out OUT
TNCA_BACKEND=sim-file $TORCH_PYTHON $T/tools/performance_harness/fp8_blockwise_cases.py --out OUT --quick
```
