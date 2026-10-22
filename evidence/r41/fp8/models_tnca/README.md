# R41 FP8: Qwen3-VL-2B-FP8 and Qwen3.5-4B-FP8 on `tnca`, layer by layer

Branch `r41-sw`, 2026-10-22. All runs used the SystemC/TLM model (`systemc-tlm-grid`) on GB10. No board or FPGA host
was used. Cycle counts are uncalibrated model predictions; this README makes numeric claims only.

## 1. Integration (r41-sw)

| step | commit | how | gate on that tree |
|---|---|---|---|
| base | `5adbd51` (r41-fp8) | – | – |
| `origin/r41-addr64` | `daf54c1` | merge, clean | see `x8` row |
| `origin/r41-tlm8` | `5afd379` | merge, clean | see `x8` row |
| `origin/r41s2-x8`, software and model parts | `e0274ce`..`aba8e5f` | cherry-pick of `0e1fd1e` (RTL: 1-byte transposed write, `HAS_XPOSE=2`; applies cleanly to the r41s1 RTL), `28d469e` (SystemC model), `a947648` (ISA sec 9E, capability `tensor_transpose_8bit`, profile `core-vrf-sync-v5-x8`, compiler), `545a55e` and the two X8 evidence commits. The Stage 2 RTL line underneath (r41s2-int) was **not** taken. `doc/support_matrix.md` was regenerated. | **gate `x8`** |
| `origin/r41-fp8-ops` `0cb5862` (B1–B6) | `86ba323` | merge. Conflict in `wide.py`: keep addr64's 64-bit division/affine code and add fp8-ops' int64→f32 conversion. Conflict in `frontend._hoisted_tile_descriptor`: keep addr64's constant-origin rebuild and add fp8-ops' guard that an entry base must be defined in the prologue (B4) | **gate `final`** |
| test fix | `39edb5e` | `test_fp8_mma.test_deep_k_split_as_elementwise_is_wrong_on_rtl`, a negative control, failed already on r41-fp8 `5adbd51` wherever the RTL bench exists (it was skipped elsewhere): the patched-out elementwise form is now refused with `Unsupported` instead of compiling wrong code. The test accepts the refusal. | `final` |
| X8 in the extended lowering | `13ebed7` (merges `r41-sw-x8dot` `14ef197`, `f7d4f1c`) | `_swapped_dot` switches to the transposed write when the target has `tensor_transpose_8bit` (section 5) | **gate `x8dot`** |

Gate (`gate/reg.sh`, `gate/rtl.sh` in the archive), on a copy of the tree with the RTL bench built:

| check | `x8` (`aba8e5f`) | `final` (`39edb5e`) | `x8dot` (`13ebed7`) |
|---|---|---|---|
| `make compiler` | compiler-unit 1/136 FAIL: the pre-existing control test above | **PASS**: compiler-unit 141, none skipped; compiler-ir 6/6 | pending |
| `default_change_audit.py` against r41s1 (`r41-fp8-audit/run-ctrl`) | **113 unchanged, 0 changed** | **113 unchanged, 0 changed** | pending |
| `fp8_cases.py` (full) | **100/100** | **100/100** | pending |
| `fp8_kernels_cases.py` (full, profile v5) | **35/35** bitwise | **35/35** | pending |
| addr64: `test_addr64.py` with RTL execution, plus `addr64_case.py` (15 case groups, 2.25/4.25 GiB, placements above 4 GiB, negatives) | **10/10**, all 15 groups pass | **10/10**, all 15 pass | pending |
| X8 RTL (`make r41-xpose8-rtl`), `make core` | pass | (RTL unchanged) | (RTL unchanged) |
| fp8-ops suites: `std_ops_cases.py`, `triangular_solve_cases.py` | – | **27 pass + 2 refused; 12/12** (identical to the fp8-ops README) | pending |

## 2. Setup for the model runs

`tools/models/qwen_tnca_layers.py` (new) and `val/val.sh` in the archive:

- Environment: `TNCA_TLM_MEMORY_BYTES=0x200000000` (8 GiB window), `TNCA_PROFILE=core-vrf-sync-v5-x8`, `TNCA_IEEE_DIV=1`,
  `TNCA_DRAM_PATH=fast`, timed model (`TNCA_TLM_UNTIMED_COLD=0`; untimed needs `memory_model=1`).
  Model binary built from this tree: SHA256 `d4e166e5…fafba2b`. torch 2.14.0, triton 3.8.0, transformers 5.20.2 (`~/flux-venv`).
- Checkpoints, as in `evidence/r41/fp8/qwen`: Qwen3-VL-2B-Instruct-FP8 `46485250`, Qwen3.5-4B-FP8 `752f65fd`.
- Inputs are real activations. Every module/layer input is captured from the CPU W8A8 reference prefill
  (`~/qwen-work/out/{vl,q35}/ref_fp8`: the image prompt, 81 / 85 tokens).
- FP8 route: transformers' own `FP8Linear` → `kernels-community/finegrained-fp8` v4 `matmul_2d`
  (`w8a8_block_dynamic_fp8_matmul_kernel`), compiled by the TNCA Triton backend under
  `torch.compile(fullgraph=True)`. Every dump contains that kernel; it issues 32 `m.mma.async.k32.e4m3` per 128-deep
  K block.

## 3. Per-layer FP8Linear (real weights, real activations)

One layer of each FP8 linear kind. Each runs at M = 1 (decode shape: the last prefill row) and at M = prefill length.
Three comparisons:

- **order**: bitwise against `fp8_block_reference.block_dynamic`, the re-derivation in TNCA's order (one binary32
  rounding per FP8 product-add in ascending K, then `*a_s`, `*b_s`, `+acc`; IEEE division).
- **CPU ref**: bitwise against `fp8_blockwise_ref` with `dot=exact` and IEEE scale. This is the CPU reference of the
  model runs. It rounds each 128-term tile dot once, so it may differ from any fp32-summing implementation by 1 ulp.
- **ratio**: rel-L2 to the exact W8A8 value (float64), divided by the bf16 rounding of that exact value. A correct GEMM
  scores 1.0.

| model | module (layer.…) | M | K × N | order bitwise | CPU ref | ratio | s (incl. compile) |
|---|---|---|---|---|---|---|---|
| VL | 13.self_attn.q_proj | 1 / 81 | 2048 × 2048 | yes / yes | yes / 1 of 165,888 (1 ulp) | 1.0000000 / 1.0000000 | 23.9 / 44.5 |
| VL | 13.self_attn.k_proj | 1 / 81 | 2048 × 1024 | yes / yes | yes / yes | 1.0 / 1.0 | 1.6 / 12.4 |
| VL | 13.self_attn.v_proj | 1 / 81 | 2048 × 1024 | yes / yes | yes / 2 of 82,944 (1 ulp) | 1.0 / 1.0 | 1.5 / 13.1 |
| VL | 13.self_attn.o_proj | 1 / 81 | 2048 × 2048 | yes / yes | yes / yes | 1.0 / 1.0 | 3.1 / 24.7 |
| VL | 13.mlp.gate_proj | 1 / 81 | 2048 × 6144 | yes / yes | yes / 3 of 497,664 (1 ulp) | 1.0 / 1.0 | 9.3 / 75.0 |
| VL | 13.mlp.up_proj | 1 / 81 | 2048 × 6144 | yes / yes | yes / 4 of 497,664 (1 ulp) | 1.0 / 1.0 | 9.3 / 73.9 |
| VL | 13.mlp.down_proj | 1 / 81 | 6144 × 2048 | yes / yes | yes / 6 of 165,888 (1 ulp) | 1.0 / 1.0 | 9.0 / 72.1 |
| Q3.5 | 14.linear_attn.in_proj_qkv | 1 / 85 | 2560 × 8192 | yes / yes | yes / 5 of 696,320 (2 ulp) | 1.0 / 1.0 | 35.4 / 140.0 |
| Q3.5 | 14.linear_attn.in_proj_z | 1 / 85 | 2560 × 4096 | yes / yes | yes / 2 of 348,160 (1 ulp) | 1.0 / 1.0 | 7.5 / 59.6 |
| Q3.5 | 14.linear_attn.out_proj | 1 / 85 | 4096 × 2560 | yes / yes | yes / 4 of 217,600 (1 ulp) | 1.0 / 1.0 | 29.2 / 82.0 |
| Q3.5 | 15.self_attn.q_proj | 1 / 85 | 2560 × 8192 | yes / yes | yes / 1 of 696,320 (1 ulp) | 1.0 / 1.0 | 14.9 / 118.9 |
| Q3.5 | 15.self_attn.k_proj | 1 / 85 | 2560 × 1024 | yes / yes | yes / yes | 1.0 / 1.0 | 1.9 / 14.8 |
| Q3.5 | 15.self_attn.v_proj | 1 / 85 | 2560 × 1024 | yes / yes | yes / yes | 1.0 / 1.0 | 1.8 / 14.6 |
| Q3.5 | 15.self_attn.o_proj | 1 / 85 | 4096 × 2560 | yes / yes | yes / 3 of 217,600 (2 ulp) | 1.0 / 1.0 | 7.3 / 59.1 |
| Q3.5 | 15.mlp.gate_proj | 1 / 85 | 2560 × 9216 | yes / yes | yes / 3 of 783,360 (1 ulp) | 1.0 / 1.0 | 16.8 / 134.2 |
| Q3.5 | 15.mlp.up_proj | 1 / 85 | 2560 × 9216 | yes / yes | yes / 4 of 783,360 (31 ulp\*) | 1.0 / 1.0 | 16.7 / 134.7 |
| Q3.5 | 15.mlp.down_proj | 1 / 85 | 9216 × 2560 | yes / yes | yes / 6 of 217,600 (2 ulp) | 1.0 / 1.0 | 16.6 / 132.5 |

**Result: 34/34 pass.**

- Every output equals the TNCA-order re-derivation bit for bit.
- Every M=1 output equals the CPU reference bit for bit.
- At prefill, at most 6 elements per layer differ from the CPU reference's exact-dot rounding.
- The worst ratio to the bf16 rounding of exact is 1.0000000083; the CPU reference itself scores the same.
- \*The 31-ulp element is a value near zero after cancellation. There, fp32 summation order (TNCA, any GPU) and the
  single rounding of the exact dot differ in the low bits; the device value is the TNCA-order value.
- Ratios are rounded to 1.0 in the table. Raw values are in `val/*/result.json`.
- Qwen3.5's `in_proj_a/b` are BF16 in the checkpoint, not FP8, so they are not FP8Linear.

## 4. Full decoder layers and a vision block on tnca

The official transformers modules run under `torch.compile(fullgraph=True)`, with everything on tnca: RMSNorms,
M-RoPE, attention or gated DeltaNet, SiLU MLP, residuals, and the FP8Linears through the Hub kernel.

- The input is teacher-forced: the CPU reference's input hidden state for that layer. The captured kwargs (position
  embeddings, ids) come from a cache-free CPU prefill. That run's layer input is bitwise equal to the reference's
  hidden state, and its CPU layer output is bitwise equal to the reference's next hidden state.
- The comparison is against the same layer on CPU with the W8A8 reference route.
- The **floor** is the CPU's own sensitivity to one rounding change: the reference with CUDA's reciprocal activation
  scale (`scale=recip`). Dynamic FP8 activation quantization turns ulp changes into code flips. `dot=fp32` moves the
  layer by ≤ 4.4e-6.

| layer | tokens | device vs CPU W8A8 layer, rel-L2 (cosine) | floor (`scale=recip`) | kernels | wall | result |
|---|---|---|---|---|---|---|
| Qwen3-VL text layer 13 (full attention) | 81 | **9.27e-4** (0.9999996) | 6.87e-4 | 16 | 5.9 min | pass |
| Qwen3.5 layer 15 (full attention, output gate) | 85 | **1.17e-2** (0.99993) | 1.05e-2 | 21 | 12.3 min | pass |
| Qwen3.5 layer 14 (**linear attention**: gated DeltaNet with conv1d, cumsum, softplus, `linalg_solve_triangular` decomposition, chunked bmm) | 85 | **1.09e-2** (0.99994) | 9.22e-3 | 31 | 14.9 min | pass |
| Qwen3-VL vision block 0 (BF16, `fullgraph=True`, `capture_scalar_outputs` for `cu_seqlens`) | 256 patches | **3.39e-3** (0.999994) | – | 18 | 3.1 min | pass |

- Pass criterion: finite, and rel-L2 ≤ max(2 × floor, 1e-2). The vision block criterion is rel-L2 ≤ 1e-2.
- Every device result sits at about 1.1–1.35× the floor. That is what a correct device graph gives: its non-FP8 ops
  (attention, norms, exp/log1p/sin/cos) round differently from CPU, and those differences become activation-code flips.
- **The Qwen3.5 linear-attention layer is no longer blocked.** With r41-fp8-ops merged, B1–B6 all lower on the
  standard path: 31 kernels, including the triangular-solve decomposition and the `cumsum`/`softplus` reduction.
  There are no host fallbacks.
- Peak RSS: 7.1 GiB for Qwen3.5 and 4.0 GiB for Qwen3-VL.

## 5. X8 transposed write in the extended lowering (`r41-sw-x8dot`)

When the target has `tensor_transpose_8bit`, `_swapped_dot`'s case (a computed A, and W read K-contiguously as
`b[k, n]`) becomes `C = A B`:

- one `t.ld.async.b.t` per K step and per column chunk, from a TensorMap over W's rows;
- A stored row-major into its window, with no scatter;
- C read out of the accumulators row by row, with no transposed scatter.

The column index is data (`offs_bn % N`), so a run-time check per chunk marks it affine or wrapping. A wrapping chunk
gathers its rows into scratch and is transferred plainly. The products and the ascending-K order are the same as before,
bit for bit. Targets without the capability keep the existing code.

- `fp8_kernels_cases.py` under `core-vrf-sync-v5-x8`: **35/35 bitwise**. The block-dynamic kernel now contains 32
  transposed 1-byte loads.
- Under `core-vrf-sync-v5`, base and candidate produce byte-identical assembly (6 kernels compared).
- Model device cycles, base gather → X8 (uncalibrated):

| case | base | X8 dot | speedup |
|---|---|---|---|
| block-dynamic 1×256×256 | 248,664 | 197,034 | 1.26 |
| block-dynamic 7×136×144 | 209,587 | 216,052 | 0.97 |
| block-dynamic 16×256×256 | 269,275 | 227,719 | 1.18 |
| block-dynamic 129×200×328 | 5,289,585 | 5,404,608 | 0.98 |
| block-dynamic 7×6144×2048 | 32,742,230 | 29,958,885 | 1.09 |
| block-dynamic 16×9216×2560 | 60,846,639 | 53,035,652 | 1.15 |
| block-dynamic 129×2048×2048 | 180,872,212 | 164,355,247 | 1.10 |
| block-static 16×2048×2048 | 10,136,972 | 7,677,488 | 1.32 |
| tensor-dynamic 16×512×2048 | 3,601,988 | 2,782,356 | 1.29 |

The two shapes with ragged N and wrapping chunks (7×136, 129×200) are about 2–3 % slower, because of the run-time
check and the scratch gather. Real layer shapes gain 9–15 %.

Model layers under the X8 dot: pending.

## 6. Remaining FP8 gaps

1. **No full-model run on tnca.** One 5.5 GiB read takes about 17 min on the model. Prefill and decode of the whole
   model (28 / 32 layers plus the embedding and lm_head) were not run. Layer-level evidence stands in for it.
2. **Teacher-forced decode with the KV/state cache on device** was not run as a layer. Decode-shape FP8 GEMMs (M=1)
   were run.
3. **Activation-scale rounding.** `TNCA_IEEE_DIV=1` reproduces the literal `amax/448`. CUDA's Hub kernel computes
   `amax*fl(1/448)`, which moves a layer by about 1e-2. That is the floor above, not an error, but it means TNCA and
   GPU W8A8 outputs are not bitwise comparable.
4. **`torch._scaled_mm` blockwise (v2)** still raises `NoValidChoicesError` on tnca: the TMA templates are not enabled
   for tnca descriptors (kernels README).
5. **Vision dynamic shapes.** The vision block needs `capture_scalar_outputs=True` for `cu_seqlens.tolist()`. The
   full-tower graph's unbacked symbols (B4 of the Qwen README) were not exercised beyond one block.
6. **X8 is a Stage 2 hardware feature.** Its RTL was cherry-picked onto the r41s1 RTL and passes the X8 RTL tests and
   `make core`. Only the 8 GiB SystemC model runs it; no board image has it.
7. The packed FP8 MAC (r41-pmac) is not used. `_mma_into` and the X8 dot issue one MMA command per K32 step.

## Archive (`models-tnca-evidence.tgz`)

| path | contents |
|---|---|
| `val/<run>/result.json`, `log`, `asm/` | per run: every comparison, knobs, timing; the `.s` and `.json` of each compiled kernel. Runs: `vl-lin`, `q35-lin`, `vl-layer13`, `q35-layer15`, `q35-layer14`, `vl-vision0` |
| `val/val.sh` | the launcher (environment above) |
| `gate/{x8,final}/` | status, logs, the default-audit comparison, fp8 / kernels / addr64 reports |
| `gate/reg.sh`, `gate/rtl.sh` | the gate scripts |

Reproduce on GB10: `val.sh NAME CORES linears|layer|vision --model vl|q35 --teacher-force ~/qwen-work/out/{vl,q35}/ref_fp8 …`
(see `tools/models/qwen_tnca_layers.py --help`).
