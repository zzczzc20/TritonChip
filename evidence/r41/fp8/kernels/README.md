# The Hub FP8 kernels and transformers' FP8Linear on `tnca`

Branch `r41-fp8`. Development evidence: SystemC/TLM model runs (`systemc-tlm-grid`,
uncalibrated cycle predictions); no board was used.

The kernels are `kernels-community/finegrained-fp8`, unchanged:

- **v4** (`version=4`, the build transformers 5.20.2 loads; module
  `_finegrained_fp8_cuda_fe2bb66`, snapshot `0e5764aa`):
  - `fp8_act_quant`;
  - `matmul_2d` with its three kernels: block-dynamic, which fuses the
    act-quant per K block; block-static; and tensor-dynamic.
- **v1**: the repository's `main` branch, `0fc209e0`. This is the build with
  `act_quant.py` and `matmul.py`. The blob hashes were checked against the Hub
  API. It is loaded with `kernels.get_local_kernel`, and provides
  `fp8_act_quant`, `w8a8_block_fp8_matmul` and `w8a8_tensor_fp8_matmul`.

The TNCA Triton backend compiles their Triton source. Nothing in the compiler
names a kernel, a model or a layer.

## Result

`tools/performance_harness/fp8_kernels_cases.py`, full run on the final tree:
**35 pass, 0 fail, 0 error**. Every output is compared bit for bit against
`fp8_block_reference.py`, a plain-torch re-derivation in the order TNCA executes:

- per K block, one binary32 rounding per FP8 product-add, in ascending K, from +0.0;
- then `* a_s`, then `* b_s`, then `+ acc`, each rounded once (TNCA does not contract to an FMA);
- **IEEE division**.

| Group | Case (M×N×K, bf16 in/out) | Result |
|---|---|---|
| v4 `fp8_act_quant` (block 128) | 4×256, 7×384, 129×256, 16×2048, 3×6144 | values and scales: 0 mismatches |
| v4 block-dynamic | tails: 1×256×256, 7×136×144, 16×256×256, 33×256×384, 129×200×328 | 0 mismatches |
| v4 block-dynamic | real layer shapes: 7×6144×2048, 7×2048×6144, 16×9216×2560, 129×2048×2048 | 0 mismatches. Model cycles: 32.5 M, 34.0 M, 60.5 M, 180 M |
| v4 block-static | 7×136×144, 129×200×328, 16×2048×2048 | 0 mismatches |
| v4 tensor-dynamic (`fp8_act_quant` over the whole K row, then the autotuned GEMM) | 7×256×256, 129×200×328, 16×512×2048 | 0 mismatches |
| v1 `fp8_act_quant` | 7×384, 129×256 | 0 mismatches |
| v1 `w8a8_block_fp8_matmul` | 7×136×256, 33×256×384, 129×200×384, 7×6144×2048 | 0 mismatches |
| v1 `w8a8_tensor_fp8_matmul` | 7×256×256, 129×200×384 | 0 mismatches |
| `torch.compile(fullgraph=True)` | block-dynamic ×2 (fused with an Inductor epilogue), tensor-dynamic, `fp8_act_quant` | 0 mismatches. Inductor traces the `triton_op` and runs the Hub kernel as a user-defined Triton kernel |
| transformers `FP8Linear` | eager x(1,7,384)→320; compiled; eager x(1,16,2048)→2048 | 0 mismatches. Calls go `FP8Linear.forward → fp8_linear → finegrained_fp8_linear → matmul_2d → w8a8_block_dynamic_fp8_matmul_kernel` on tnca |

The tails are:

- M = 1, 7, 33 and 129 (block M of 16, 64 and 128);
- N and K that are not multiples of 128: 136, 144, 200 and 328;
- a 3-D input.

## Numerics: division

The kernels compute `amax / 448.0` and `x / scale`. Their source semantics is
IEEE division.

**TNCA follows the IEEE semantics.** The new `mathlib.div_ieee` takes the
faithful quotient and applies one residual correction. That correction is
exact in one FMA, and the result is correctly rounded. It was checked against
exact rational division on 20 000 random pairs: 0 differences, against 5 176 for
the faithful quotient.

- The extended lowering, which compiles every Hub matmul, always uses it.
- Kernels that compile through the default lowering (`fp8_act_quant`) use it
  under the new `ieee_div` option (`TNCA_IEEE_DIV=1`). The harness sets this
  option.

**TNCA's default `/` is not IEEE-exact.** It is faithful, within 1 ulp
(`mathlib.div`, `ACCURACY['div']`), and this is unchanged so that the frozen
kernels keep their code. Without the option, `fp8_act_quant` differs from the
IEEE re-derivation in:

- about half the scales, by 1 ulp;
- 5 of 1 024 FP8 codes at 4×256.

**CUDA is different.** On CUDA, Triton's default fp32 divide makes `amax / 448`
equal `amax × RN(1/448)`. This is a property of the platform, not of the kernel
source; the runners agent measured it (`evidence/r41/fp8/qwen`):

- about 0.1 % of activation codes flip;
- layers move by up to 5e-3.

TNCA does not reproduce this.

## Generated code (`runs/final`, `asm/` in the archive)

For `w8a8_block_dynamic_fp8_matmul_kernel` at block M = 16, the code is 13 554
instructions; the code store holds 16 384. The K loop does the following for
each 128-deep K block:

1. **Act-quant.** The vector unit computes the act-quant: bf16 row loads, a
   lane-wise `vf.max` fold over the eight 16-lane chunks, then
   `vf.reduce.max.b` per row. `a_s = amax/448` is computed by `div_ieee` with
   `v.rcp.approx`, a Newton step and the correction. `vf.mul` by the
   reciprocal of `max(a_s, 1e-12)` follows, then `vf.cvt` to e4m3 with the
   FP8 NaN rule.
2. **FP8 MMA.** There are 32 `m.mma.async.k32.e4m3` per 128-deep K block: 8
   column chunks × 4 K32 steps. Each is accumulated into a freshly zeroed
   accumulator, in ascending K. The MMA is the product transposed:
   - B^T is the A operand. Each 32-byte K run of a weight row W[n, k] is one
     predicated `v.ld.w`, with a byte mask for the K tail, written straight
     into window registers;
   - the quantised activation, scattered with `v.st.scatter.b`, is the B operand.

   Every product is exact and every column sums in ascending K, so Cᵀ is C
   transposed bit for bit. The 16 rows of Cᵀ are scattered back transposed into
   scratch with 16 `v.st.scatter.w`.
3. **Per-block scaling.** Scaling runs in the vector unit, row by row:
   - `vf.mul` by the row's `a_s` (a lane-0 extract and splat);
   - `vf.mul` by the 16-lane `b_s` chunk vector;
   - `vf.add` into the loop-carried tile.

   That is 384 accumulator reads and about 385 `vf.mul` per block of code.
   There are no `a.mul.row` or `a.fma.row`.
4. **Large M.** At block M above 16 (M ≥ 17), `row_split` emits the same body
   once per 16-row chunk inside a loop.

The per-K `m.mma` path is unchanged. The packed-MAC opcode can come in as an
alternative to the step in `_swapped_dot` and `_mma_into`, which issue one
command per K step.

## What changed (all generic; default code unchanged)

A kernel that the default lowering refuses, or whose code overflows the code
store, is imported again with the importer's `large_tiles` set (`aot._lower_ttir`). Under it:

- **Separable indices** (`tnca_cc/separable.py`). For `offs % M` rows, runtime
  strides and column gathers, row r is one scalar address and one lane vector.
  A carried 2-D pointer carries a single shared scalar delta.
- **K-split native dots** (`register_blocks._native_matrix`, `frontend._mma_fragments`).
- **The swapped FP8 dot** (`frontend._swapped_dot`), as described above. It
  includes a row-bound analysis of the mask (`_row_limit`).
- **`row_split.py`**, which runs independent rows as a loop over 16-row chunks.
  A union-find over tensor dimensions finds a row dimension that no reduction,
  dot K or reshape couples. The transform is refused if a load follows a store.
- **Wide 1-D blocks**, up to 16 384. Max/min reductions fold lane-wise and
  then reduce once. Register-chunk starts stay as constants.

Other changes:

- **Resource numbering by lifetime when the fixed numbering overflows.**
  - Barriers, accumulators and window ids: `tnca_cc/resource_lifetime.py`, used from `codegen.assign_shared`.
  - Window registers: a fallback to `window_lifetime` in `aot._emit`.
  - Token registers: recycled when scalars run out (`codegen._alloc`).
- **IEEE division**: `mathlib.div_ieee` and the `ieee_div` option.
- **Autotune.**
  - `triton.autotune` on tnca compiles each config without running it
    (`driver.bench`, `TNCA_LAUNCH=skip` semantics).
  - A config tnca cannot lower raises `OutOfResources`, so the tuner drops it.
    The tensor kernel's BN=256 configs are dropped this way.
  - Warps and stages no longer split the compile cache.
- **Bug fixes that apply everywhere.**
  - Loop-carried 1-D pointer tensors lost their per-lane offsets: a silent
    miscompile. Every lane loaded lane 0's address.
  - Dots that are not native matrix operations, with M > 16 or K > 16, were
    split elementwise. They are now refused.

## The FP8Linear integration point

- **Device context.** `utils.device_context` calls `getattr(torch, 'tnca')`.
  That is the existing device module, which has no `.device` attribute, so the
  context is a no-op. Nothing was needed.
- **Kernel build.** The `kernels` package picks `torch-cuda` because
  `torch.version.cuda` is set. That build is pure Triton, and the active Triton
  driver is TNCA's (with `TNCA_TRITON=1` or no CUDA device).
- **DeepGEMM** is not loadable without CUDA, so `fp8_linear` takes the Triton
  route.
- **Blocker: transformers' quantizer.**
  `FineGrainedFP8HfQuantizer.validate_environment` checks for a device. When
  `not torch.cuda.is_available() and not is_torch_xpu_available()` and the
  checkpoint is pre-quantised, it sets `quantization_config.dequantize = True`
  and logs a warning:
  - this was measured: `dequantize` is True after the call;
  - no config field prevents it. `dequantize=False` is overwritten;
  - `register_quantizer` refuses a name that is already registered.
- **Minimal generic mechanism that keeps FP8 weights.** Use transformers' own
  public replacement, `integrations.finegrained_fp8.replace_with_fp8_linear(model,
  quantization_config=FineGrainedFP8Config(...), pre_quantized=True)`. Then
  load the checkpoint's `weight` and `weight_scale_inv` tensors with
  `load_state_dict` and call `.to('tnca')`. The harness's transformers rows use
  exactly this.
- **The alternative is a registry replacement, which is not a documented API.**
  A subclass of the quantizer that accepts the PrivateUse1 device would be
  installed in `AUTO_QUANTIZER_MAPPING['fp8']`.

## Not done

**Blockwise `torch._scaled_mm`, the second route,** did not work when this was written: CPU eager refuses
blockwise scales, and Inductor's only lowering is a CUDA-TMA template (`NoValidChoicesError` on tnca,
`runs/scaled_mm_v2_blockwise_probe.log`). **It has since been done:** see `evidence/r41/fp8/blockwise/README.md`,
a tnca K-block template, bitwise against plain-torch math in device order.

## No regression (final tree `files:31db6d37…`, commit 0b285e6)

| Check | Result |
|---|---|
| `default_change_audit.py`, compared with r41s1 | **113 unchanged, 0 changed** |
| `fp8_cases.py` (full, `systemc-tlm-grid`) | **100/100 pass** |
| `make compiler` | generated-file checks pass; `compiler-unit` 126 tests pass, 25 skipped (no RTL bench in this environment); `compiler-ir` 6/6 pass |

## Archive (`fp8-kernels-evidence.tgz`)

| Path | Contents |
|---|---|
| `runs/final/` | `report.json` (per case: comparisons, launches, device cycles, code facts) and the harness log |
| `runs/quick-report.json` | the quick run |
| `asm/<kernel>/` | the `.s` and `.ttir` of each compiled kernel |
| `regression/` | audit JSON and logs, the `fp8_cases` report and log, the `make compiler` log |

## Reproduce (GB10)

```sh
T=~/r41-fp8-kern
CUDA_VISIBLE_DEVICES= HF_HOME=~/flux-hf TNCA_TLM_BINARY=~/r41-fp8-build/tnca-chip \
  ~/flux-venv/bin/python $T/tools/performance_harness/fp8_kernels_cases.py --out OUT \
  --v1-path DIR_WITH_main_build [--quick] [--group v4|v1|compile|transformers]
```
