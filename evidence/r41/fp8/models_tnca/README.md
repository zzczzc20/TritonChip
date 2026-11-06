# R41 FP8: Qwen3-VL-2B-FP8 and Qwen3.5-4B-FP8 on `tnca`, layer by layer

Branch `r41-sw`, 2026-10-22. Sections 6 and 7 (decode with the cache on the device, the vision tower) were added after
the merge of r41-fp8-ops V1–V5. All runs used the SystemC/TLM model (`systemc-tlm-grid`) on GB10. No board or FPGA host
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
| `origin/r41-fp8-ops` `63dd9e6` (V1–V5: int64 loop-carried values, erf, host metadata for `cu_seqlens`, the `mask_index` device path, the V1 descriptor rule) | `d271d60` | merge. Conflict in `wide.py`: fp8-ops' `divide64` duplicates addr64's exact 64-bit division. Kept addr64's `_divide` (constant divisor) and `_divide_dynamic` (any divisor), and routed a 64-bit **tensor** division by a non-constant divisor (fp8-ops' new case) to `_divide_dynamic`; added fp8-ops' `carried_pair`. Conflict in `frontend` `scf.yield`: kept addr64's indexed `values[i]` (separable carries, shared scalars) and added fp8-ops' wide (int64) carries in that form. `doc/support_matrix.md` regenerated (`--check` clean) | **gate `v15`** |

Gate (`gate/reg.sh`, `gate/rtl.sh` in the archive), on a copy of the tree with the RTL bench built:

| check | `x8` (`aba8e5f`) | `final` (`39edb5e`) | `x8dot` (`13ebed7`) | `v15` (`d271d60`) |
|---|---|---|---|---|
| `make compiler` | compiler-unit 1/136 FAIL: the pre-existing control test above | **PASS**: compiler-unit 141, none skipped; compiler-ir 6/6 | **PASS**: 141 / 6 | **PASS**: 143 / 6 |
| `default_change_audit.py` against r41s1 (`r41-fp8-audit/run-ctrl`) | **113 unchanged, 0 changed** | **113 unchanged, 0 changed** | **113 unchanged, 0 changed** | **113 unchanged, 0 changed** |
| `fp8_cases.py` (full) | **100/100** | **100/100** | **100/100** | **100/100** |
| `fp8_kernels_cases.py` (full, profile v5) | **35/35** bitwise | **35/35** | **35/35** | **35/35** |
| addr64: `test_addr64.py` with RTL execution, plus `addr64_case.py` (15 case groups, 2.25/4.25 GiB, placements above 4 GiB, negatives) | **10/10**, all 15 groups pass | **10/10**, all 15 pass | **10/10**, all 15 pass | **10/10** (TLM and RTL), all 15 pass |
| X8 RTL (`make r41-xpose8-rtl`), `make core` | pass | (RTL unchanged) | (RTL unchanged) | pass (rerun) |
| fp8-ops suites: `std_ops_cases.py`, `triangular_solve_cases.py` | – | **27 pass + 2 refused; 12/12** (identical to the fp8-ops README) | **27 + 2 refused; 12/12** | **48 pass + 1 refused; 12/12** (identical to the fp8-ops V1–V5 README) |

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

Model layers under the X8 dot (`val/vl-lin-x8dot`, `val/vl-layer13-x8dot`): the 14 Qwen3-VL linears of section 3 again pass 14/14. They are bitwise equal to the TNCA-order re-derivation, with the same 16 prefill elements off the exact-dot CPU reference. The kernel now has 32 `t.ld.async.b.t`. Decoder layer 13 again gives rel-L2 9.27e-4 against the CPU layer, the same mismatch count as before. Wall times are not comparable: the gate shared the cores.

## 6. Decode with the KV cache / linear-attention state on tnca

`qwen_tnca_layers.py decode` (runs `val/vl-decode`, `val/q35-decode`, same environment as section 2, branch tree
`d271d60` + `eb6709a`). Two consecutive decoder layers run as one module under `torch.compile(fullgraph=True)`.
Their weights and transformers' own `StaticCache` live on tnca:

- **Qwen3-VL-2B-FP8, layers 13 + 14** (both full attention): a static KV cache of 32 slots.
- **Qwen3.5-4B-FP8, layers 14 + 15**: layer 14 is linear attention (gated DeltaNet). Its conv state is bf16
  `[1, 8192, 4]` and its recurrent state is **fp32** `[1, 32, 128, 128]`. Layer 15 is full attention with a static
  KV cache of 32 slots.

How the run works:

- **Prompt.** A short text prompt, "Name three primary colors.", through the official chat template: 13 tokens
  (VL) and 17 tokens (Q3.5). The run is the prefill plus **4 decode steps**, with a fixed bucket of 32.
- **Graphs.** One prefill graph and one decode graph. The decode graph is compiled at decode step 1. Before steps 2–4,
  `torch._dynamo.config.error_on_recompile` is set, so any recompile would fail the step. There were none:
  `unique_graphs` grew by 0 at steps 2–4.
- **Teacher forcing.** The CPU W8A8 reference is the whole model with a static cache: prefill plus greedy decode,
  route `dot=exact`, IEEE scale. At every call, hooks capture the first layer's input hidden state and both layers'
  kwargs (position embeddings, mask, position ids), outputs and cache contents. The device pair gets the CPU input
  of the first layer at every step. The second layer gets the device output of the first. **Each cache is only ever
  written by its own device run.**
- **Harness check.** The same pair, run eagerly on CPU with its own cache, reproduces the whole-model run's outputs
  and cache contents **bit for bit** at every step, for both models.
- **Linear-attention states.** They are allocated on the device before the prefill (zeros, through the cache
  layer's own `lazy_initialization`, with the CPU state's dtype). The prefill graph then writes them. The recurrent
  state is fp32 on the device after every step (`state_dtypes`), so it was not downcast.
- **Floors.** These are the CPU reference's own rounding floor, as in section 4, now three runs of the same pair:
  - `scale=recip`;
  - `dot=fp32`, which is bitwise equal to the reference here: 0 for every tensor;
  - **CPU Inductor**: the same pair compiled for the host. Fused pointwise ops are computed in fp32 and rounded once,
    as on any Inductor backend, instead of eager's bf16 rounding after every op.

  The pass criterion, per step and per tensor: finite, same dtype, and rel-L2 ≤ max(2 × the largest floor, 1e-2).

Results: rel-L2 of the device against the CPU W8A8 run. Floors are given in parentheses as `scale=recip` / CPU
Inductor. "decode, worst" is the largest rel-L2 over steps 1–4.

| model | tensor | prefill | decode, worst | min cosine | 5/5 steps |
|---|---|---|---|---|---|
| VL | layer 13 output | 3.34e-4 (2.59e-4 / 3.27e-4) | 1.42e-2 (1.05e-2 / 1.29e-2) | 0.99990 | pass |
| VL | layer 14 output | 5.42e-4 (4.93e-4 / 5.38e-4) | 2.18e-2 (2.02e-2 / 2.06e-2) | 0.99976 | pass |
| VL | layer 13 K cache | 7.68e-3 (2.30e-3 / 7.90e-3) | 7.74e-3 (2.22e-3 / 7.92e-3) | 0.99997 | pass |
| VL | layer 13 V cache | 5.03e-3 (1.39e-3 / 5.15e-3) | 5.37e-3 (1.43e-3 / 5.51e-3) | 0.99999 | pass |
| VL | layer 14 K cache | 1.64e-2 (1.47e-2 / 1.71e-2) | 1.63e-2 (1.46e-2 / 1.69e-2) | 0.99987 | pass |
| VL | layer 14 V cache | 1.33e-2 (1.18e-2 / 1.33e-2) | 1.51e-2 (1.36e-2 / 1.52e-2) | 0.99989 | pass |
| Q3.5 | layer 14 output (linear attn) | 9.51e-3 (8.48e-3 / 9.96e-3) | 1.46e-2 (1.21e-2 / 1.33e-2) | 0.99990 | pass |
| Q3.5 | layer 15 output | 1.78e-2 (1.67e-2 / 1.85e-2) | 2.46e-2 (2.42e-2 / 2.47e-2) | 0.99970 | pass |
| Q3.5 | layer 14 conv state (bf16) | **0, bitwise** (1.54e-3 / 7.18e-3) | **0, bitwise** (1.55e-3 / 7.78e-3) | 1.0 | pass |
| Q3.5 | layer 14 recurrent state (**fp32**) | 4.16e-3 (2.84e-3 / 8.44e-3) | 4.70e-3 (1.83e-3 / 1.01e-2) | 0.99999 | pass |
| Q3.5 | layer 15 K cache | 1.82e-2 (1.80e-2 / 2.08e-2) | 1.84e-2 (1.77e-2 / 2.06e-2) | 0.99983 | pass |
| Q3.5 | layer 15 V cache | 2.24e-2 (2.09e-2 / 2.37e-2) | 2.25e-2 (2.10e-2 / 2.37e-2) | 0.99975 | pass |

**Result: both pairs pass all 5 steps.**

- Every tensor's ratio to the largest floor is 0.0–1.10.
- No error grows over the steps. The layer 13 K cache, for example, is 7.68e-3 after the prefill and 7.67e-3 after
  step 4.
- Against the `scale=recip` floor alone, two tensors are well above it:
  - The first-layer VL KV cache is 3.3–3.8× that floor. The CPU Inductor floor, 7.9e-3, is where it sits: these are
    the k/v projections after the rotary and k-norm pointwise ops, which Inductor rounds once and eager rounds per
    op.
  - The Q3.5 recurrent state is 1.5–2.6× the `scale=recip` floor. It is at 0.3–0.5× the CPU Inductor floor.
- The conv state is bitwise equal to the CPU reference at every step.
- A second, identical run of each model (`val/*-decode-1`, criterion of an earlier harness revision) gave the same
  device rel-L2 to every digit.

The state stays on the device. A meter wraps every host read and write of tnca memory (`tnca.torch_device.MEMORY`:
`to_host`, `from_host`, `fill_`, `.item()`) and the runtime launch log (per-launch staging bytes). It is open from the
call into the compiled pair until the device is drained. Inputs are moved before the call and outputs read back after
it. Cache contents are read for the comparison after the meter closes.

| model | step | wall s | launches | staging up / down | host reads / writes inside the step | resident bytes | model device cycles |
|---|---|---|---|---|---|---|---|
| VL | prefill (compiles) | 90.8 | 75 | 0 / 0 | 8 / 34 calls, 26 / 272 B | 108,380,184 | 259,448,092 |
| VL | decode 1 (compiles) | 78.4 | 41 | 0 / 0 | 6 / 6 calls, 16 / 48 B | 101,659,648 | 225,878,719 |
| VL | decode 2, 3, 4 | 61.6–61.7 | 41 each | 0 / 0 | **0 / 0** | 101,659,648 | 225.55–225.71 M |
| Q3.5 | prefill (compiles) | 413.2 | 124 | 0 / 0 | 14 / 36 calls, 50 / 288 B | 318,419,320 | 1,211,280,402 |
| Q3.5 | decode 1 (compiles) | 166.3 | 47 | 0 / 0 | 4 / 4 calls, 12 / 32 B | 232,718,240 | 479,007,077 |
| Q3.5 | decode 2, 3, 4 | 134.8–134.9 | 47 each | 0 / 0 | **0 / 0** | 232,718,240 | 478.61–478.95 M |

- **Every launch ran on resident tensors**, with no launch staging in any step.
- **The steady decode steps make no host access at all.** The only host accesses happen in the two calls that
  compile, and each is attributed to its call site (`host_sites` in `result.json`):
  - Inductor's pattern-matcher initialisation (`fuse_attention._get_sfdp_patterns`, example tensors of 8–16 B);
  - constant folding (`joint_graph._deduce_value`).

  Neither touches the cache.
- Peak RSS was 4.0 GiB for VL and 6.9 GiB for Q3.5. Total wall time was 6.6 min and 17.7 min.


## 7. Qwen3-VL vision tower on tnca

`qwen_tnca_layers.py tower` (`val/vl-tower`). The whole tower runs as **one** `torch.compile(fullgraph=True)`
graph: the patch embedding, the interpolated position embedding, all 24 blocks, the merger and the 3 deepstack
mergers. `grid_thw` and `cu_seqlens` stay on the CPU as host metadata (V4). Rotary positions and interpolation indices
are precomputed with `transformers.vision_utils`, as in `cmd_run`.

- **Input.** The CPU runner's fixed 256×256 image: grid 1×16×16, 256 patches, 64 merged tokens.
- **Graph.** 28 kernels and 399 launches. That matches the compile-only V1–V5 probe.
- **CPU reference.** The CPU bf16 tower reproduces the reference run's `vision_embeds` and the three deepstack
  features **bit for bit**.
- **Floor.** There is no FP8 in the tower, so the floor is the CPU reference's own bf16 rounding: the same tower in
  fp32 on CPU, compared with the bf16 one.

| output | shape | device vs CPU bf16, rel-L2 (cosine) | floor (CPU bf16 vs CPU fp32) | ratio | device vs CPU fp32 | result |
|---|---|---|---|---|---|---|
| `vision_embeds` (merger) | 64 × 2048 | **4.40e-2** (0.99903) | 3.78e-2 | 1.16 | 2.20e-2 | pass |
| deepstack 0 (block 5) | 64 × 2048 | **1.07e-2** (0.99994) | 9.03e-3 | 1.18 | 6.91e-3 | pass |
| deepstack 1 (block 11) | 64 × 2048 | **1.28e-2** (0.99992) | 1.01e-2 | 1.27 | 8.54e-3 | pass |
| deepstack 2 (block 17) | 64 × 2048 | **1.80e-2** (0.99984) | 1.43e-2 | 1.26 | 1.00e-2 | pass |

- The criterion is: finite, and rel-L2 ≤ max(2 × floor, 1e-2). All 4 outputs pass, at 1.16–1.27× the floor.
- On every output, **the device is closer to the fp32 tower than the CPU bf16 reference is**, at 0.58–0.85× of the
  CPU reference's own distance. Inductor computes fused pointwise ops in fp32 and rounds once.
- Peak RSS: 4.7 GiB.
- **Wall time was 3 h 04 min, above the 2 h budget.** The estimate before the run was about 75 min: 24 × block 0's
  3.1 min (section 4), plus the merger. The real cost was about 7.7 min per block, counting the mergers. That is about 2.5× block 0
  alone, whose kernels compiled in seconds. The cause was not investigated. The run was allowed to finish, because no
  other work was waiting on those cores.

## 8. Remaining FP8 gaps

1. **No full-model run on tnca.** One 5.5 GiB read takes about 17 min on the model. Prefill and decode of the whole
   model (28 / 32 layers plus the embedding and lm_head) were not run. Layer-level evidence stands in for it.
2. **Decode with the cache on the device ran on two layers per model, not on the whole model** (section 6). The
   cache bucket was 32 slots and the prompt was short text. A long-context bucket and an image prompt in decode were
   not run.
3. **Activation-scale rounding.** `TNCA_IEEE_DIV=1` reproduces the literal `amax/448`. CUDA's Hub kernel computes
   `amax*fl(1/448)`, which moves a layer by about 1e-2. That is the floor above, not an error, but it means TNCA and
   GPU W8A8 outputs are not bitwise comparable.
4. **`torch._scaled_mm` blockwise (v2)** still raises `NoValidChoicesError` on tnca: the TMA templates are not enabled
   for tnca descriptors (kernels README).
5. **Vision dynamic shapes.** The full tower ran with one image (section 7: `capture_scalar_outputs=True`, and
   `cu_seqlens` as host metadata). Two or more images or video frames remain refused (fp8-ops V4: symbolic segment
   sizes).
6. **X8 is a Stage 2 hardware feature.** Its RTL was cherry-picked onto the r41s1 RTL and passes the X8 RTL tests and
   `make core`. Only the 8 GiB SystemC model runs it; no board image has it.
7. The packed FP8 MAC (r41-pmac) is not used. `_mma_into` and the X8 dot issue one MMA command per K32 step.

## Archive (`models-tnca-evidence.tgz`)

| path | contents |
|---|---|
| `val/<run>/result.json`, `log`, `asm/` | per run: every comparison, knobs, timing; the `.s` and `.json` of each compiled kernel. Runs: `vl-lin`, `q35-lin`, `vl-layer13`, `q35-layer15`, `q35-layer14`, `vl-vision0`, `vl-lin-x8dot`, `vl-layer13-x8dot` |
| `val/val.sh` | the launcher (environment above) |
| `gate/{x8,final,x8dot}/` | status, logs, the default-audit comparison, fp8 / kernels / addr64 reports |
| `gate/reg.sh`, `gate/rtl.sh` | the gate scripts |
| `models-tnca-decode-evidence.tgz` (second archive) | `gate/v15/` (status, logs, audit comparison, fp8 / kernels / std-ops / solve reports, addr64 case JSON; `make-core`, `test_addr64_rtl`, `xpose8-rtl` logs), `gate/{reg,rtl}.sh`, `val/val.sh` (tree `tree-val2`), `val/{vl-decode,q35-decode,vl-tower}/` (`result.json` with per-step tensors' comparisons, floors, transfer meters and call sites; `log`; `asm/`), `val/{vl,q35}-decode-1/` (the repeat runs) |
| `x8dot-kernels/` | fp8_kernels_cases reports: base and X8 dot under x8 (`base-x8`, `c7-x8`) and under v5 (`base-v5`, `cand-v5`); `cmp.py` makes the cycle table |

Reproduce on GB10: `val.sh NAME CORES linears|layer|vision|tower --model vl|q35 --teacher-force ~/qwen-work/out/{vl,q35}/ref_fp8 …`
and `val.sh NAME CORES decode --model vl --index 13 --bucket 32 --steps 4` (`--model q35 --index 14` for Qwen3.5).
(see `tools/models/qwen_tnca_layers.py --help`).
