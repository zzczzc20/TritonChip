# X8: 1-byte transposed window write (FP8 B operand)

**Status:** implemented on `r41s2-x8` (based on `r41s2-int` cf28fe3). The RTL, ISA, capability, profile,
SystemC model and compiler are all in. The 16-bit behaviour and its cycles are unchanged. In the model,
FP8 W8A8 linears run **3.0–8.7× faster**. Not routed. The unit OOC result is under "Timing and area".

## Why

`torch._scaled_mm` and an FP8 linear take B = Wᵀ, which is column-major (K-contiguous). Until now the
transfer engine's transposed write (ISA §9E `mod[5]`) handled only 16-bit elements, so the compiler built the
FP8 B operand out of gathers: 37–58 LSU instructions per MMA. This candidate extends the corner turn to
1-byte elements.

## RTL (`rtl/core/tnca_tensor.v`)

- **Parameter:** `HAS_XPOSE` / the core's `HAS_TENSOR_XPOSE` now takes three values:
  - `0`: not built;
  - `1`: 16-bit only (all existing images);
  - `2`: 16-bit and 1-byte. No new port or parameter has to be threaded through the design.
- **Mechanism:** for `t_esz == 0` (`x_b`, a constant 0 unless `HAS_XPOSE >= 2`):
  - a column block is 8 source columns, so it completes 8 destination rows, and `S_XWRITE` runs for 8 cycles instead of 4;
  - a destination row is `R` bytes (1, 2 or 4 lanes);
  - `x_data` selects byte `x_q` from each buffered unit;
  - `xbuf` (16 units), `S_XRUN` and `S_XDRAIN` are unchanged.
- **Geometry rules and refusals are identical to the 16-bit write:**
  - rank 2, `R` ∈ {4, 8, 16}, and a load only;
  - base, row stride and contiguous coordinate 8-byte aligned, and the coordinate non-negative;
  - a source row of whole 8-byte units, so `C` is a multiple of 8.
- **On a `HAS_XPOSE = 1` engine** a 1-byte transposed load still raises `DESCRIPTOR_FAULT`, and on a `0` engine `ILLEGAL_INSTRUCTION`.
- **Generic in dtype:** the engine moves bytes, so int8, e4m3 and e5m2 all work (the dtype code is not read).

## ISA, capability, profile

- **ISA §9E:** the 16-bit row of the condition table now also allows 1-byte elements under `tensor_transpose_8bit`, and a new paragraph describes them.
- **Capability `tensor_transpose_8bit`:**
  - `tools/isa/gen_target.py` emits it only where `HAS_TENSOR_XPOSE >= 2`, so every existing target file is byte-identical (`gen_target.py --check` passes);
  - `TargetInfo.has()` treats it as optional: when the key is absent it returns False, so **old images keep the gather path**;
  - `codegen` requires the capability before it emits `t.ld.async.b.t`.
- **Profile `core-vrf-sync-v5-x8`:**
  - `spec/r1_profiles.json`: v5 + `HAS_TENSOR_XPOSE=2`;
  - target `tnca-sim-vrf-v5-x8.json`;
  - bench `tb_core_r39sw5x8.vvp` (Makefile);
  - device parameters in `python/tnca/device.py`, including `/slow` and `/hbm`.
  - The Stage 2 image must declare this profile, and build with `HAS_TENSOR_XPOSE=2`, before compiled artifacts use the feature.
- **Support matrix:** regenerated, with a new row "1-byte transposed transfer".

## Compiler (minimal, generic)

`compiler/python/tnca_cc/frontend.py`:

- **`_transposed_tile_load`** (an FP8 tile read down its columns that only `tt.dot` reads). On a target with the capability it emits one `t.ld.async.b.t` instead of a gather, under these conditions:
  - K rows: a multiple of vlen, at most 256;
  - N ∈ {4, 8, 16};
  - not masked;
  - not a mixed-format dot. An e4m3×e5m2 dot converts its operands in registers, so its B stays on the gather path.
- **`_xpose_geometry_ok`** (the descriptor `tl.trans` path) accepts FP8 under the same capability, when `block[1] % 8 == 0`.

### What `r41-fp8` must pick up

- **Merge `r41s2-x8`.** The compiler part alone is commits a947648 and 545a55e, but running the feature needs the RTL and model commits too. `git merge` into `r41-fp8` is clean: tested as the local scratch branch `r41s2-x8-fp8`, which is r41s2-x8 + origin/r41-fp8 6256ff3.
- **No r41-fp8 code has to change for the default lowering.** In that merge the transposed path sits ahead of r41-fp8's lazy FP8 gather (`_fp8_mma_operand`), and it fires for the template's unmasked B tiles.
- **Still gathering:**
  - a masked B tile (an M/N/K tail);
  - a mixed-format dot;
  - the extended lowering's `Sep2D` / `_swapped_dot` path.
  - Extending `_swapped_dot` to the transposed B would be r41-fp8's change. It should take the transposed path when `target.has('tensor_transpose_8bit')`.
- **Run FP8 work with `TNCA_PROFILE=core-vrf-sync-v5-x8`.** Prepare the RTL device first with `tnca.device.prepare('core-vrf-sync-v5-x8')`.

## SystemC/TLM model

- **`include/tnca_tensor_engine.hpp`:** the corner turn is generic in the element size, through one helper `corner_turn`. In all four paths (functional, `core_pipeline` RTL sequencing, timeline, slow):
  - a block is 8/esize columns, with 8/esize write cycles in the RTL path;
  - a 1-byte row of 4 elements is written as a 4-byte lane.
- **`tnca_tensor_model.hpp`** accepts 1-byte transposed transfers. The model does not gate on the image, which matches how it already treats the 16-bit write; the capability gates the compiler.
- **`validate_dma.py`** adds 1-byte transposes: R=4/8/16, dtypes 3/7/8, a tail unit and an out-of-bounds row.
  - Result: `passed`.
  - The 16-bit `transpose-4/8/16` cases have counters identical to the base binary.
- **RTL oracle:** `rtl_oracle/xpose_oracle.py` with `tb_xpose_capture.v` captures `S_XRUN`/`S_XDRAIN`/`S_XWRITE` from `tb_tnca_core` (HAS=2) at memory latencies 1, 8 and 40, over t30 (16-bit) and t33 (1-byte). It replays the capture through the model's `core_pipeline` rule (72 column blocks):
  - **pass:** every block makes 8/esize writes, and the gap to the next block is 1 cycle, as the model has it;
  - the RTL rule `w = max(start + R, last_rsp + 1) + 1` holds exactly in every block;
  - the model's own error (RTL − model) is in the same set for both widths: {+1, +4} for 1-byte and {+1, +4, +8} for 16-bit, from the drain cycle and from out-of-bounds units the model does not walk. This is the pre-existing 16-bit behaviour; it was left unchanged so as not to disturb the M2 calibration.

## Tests

| Check | Result |
|---|---|
| `make r41-xpose8-rtl`. t33_xpose8 (224 lane assertions): R=4/8/16; int8/e4m3/e5m2; 1–3 column blocks; rows before and after the extent; a partial tail unit and wholly out-of-bounds units; a predicated-off command that leaves its window untouched and still completes. Refusals: t33 on a HAS=0 engine raises ILLEGAL (01); t34 (HAS=1, 1-byte) raises 04; t35 (R=12) raises 04; t36 (unaligned column) raises 04. t30/t32 also run on the HAS=2 engine | 9 PASS. t30 runs 3757 cycles on both HAS=1 and HAS=2 |
| `make r38-xpose-rtl` | 3 PASS, cycles unchanged |
| Base cf28fe3 vs X8: `make core core-vrf core-r39sw5 r38-xpose-rtl top`, every result line (including `cycles=`) | **identical**: 175 result lines each, 0 FAIL (`regress-*.txt` in the archive) |
| `validate_dma.py` (model) | passed, including 3 new 1-byte transposes |
| `xpose_oracle.py` (RTL capture vs model) | pass, as above |
| FP8 `_scaled_mm`/linear `--quick`, **SystemC model**, v5-x8 | 8/8 PASS, 0 bit mismatches against the per-K reference. 3 `t.ld.async.b.t` emitted |
| same, **RTL sim** (`sim-file`, device built with HAS=2) | 8/8 PASS, 0 bit mismatches; same instruction mix |
| same on v5 (gather), for comparison | 8/8 PASS; no `.b.t` |
| full `scaled_mm` + `linear` groups, model, v5-x8 | 59/59 PASS (0 fail, error or refused), 0 bit mismatches. 42 `t.ld.async.b.t`, 35 `t.ld.async.b` (the masked tail tiles) |

## What-if: FP8 W8A8 linears, gather vs X8, with and without packed-MAC V3

- **Setup:** r41-pmac's harness (`pmac_points.py`, configs `g1-pf0` = `r40-board-m2.cfg`, and `g4-pf0-pipe5` = V3: `mtx_dp_group_8=4`, pipe+1). This uses its model keys, cherry-picked cleanly onto the scratch tree (`r41s2-x8-fp8`, model built from it).
  - B40 knobs: 3 cores, resident, 3 ports, 8 outstanding; fast DRAM path, B40 map.
  - Untimed cold call, then the steady call.
- **The gather rows reproduce r41-pmac's numbers to the cycle**, for example 53,965,314. That cross-checks that the 16-bit and software paths are unchanged.
- These are uncalibrated model predictions (R40 holdout median error 21.5 %). Read only the ratios; the noise floor is about ±2 %.

| point | configuration | status | device cycles | speedup vs gather | LSU instr / MMA | MMA busy share | MMA wait share | max rel err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `bf16-m1-k2048-n6144` | gather (today) | pass | 53,965,314 | 1.00x | 58.2 | 9.2 % | 0.4 % | 0.0019 |
| `bf16-m1-k2048-n6144` | gather + V3 | pass | 54,488,853 | 0.99x | 58.2 | 2.6 % | 0.0 % | 0.0019 |
| `bf16-m1-k2048-n6144` | X8 | pass | 10,731,757 | 5.03x | 22.3 | 46.1 % | 3.2 % | 0.0019 |
| `bf16-m1-k2048-n6144` | X8 + V3 | pass | 9,782,561 | 5.52x | 22.3 | 14.6 % | 0.1 % | 0.0019 |
| `bf16-m1-k2560-n9216` | gather (today) | pass | 51,977,068 | 1.00x | 58.2 | 17.9 % | 1.6 % | 0.0018 |
| `bf16-m1-k2560-n9216` | gather + V3 | pass | 49,931,589 | 1.04x | 58.2 | 5.4 % | 0.1 % | 0.0018 |
| `bf16-m1-k2560-n9216` | X8 | pass | 17,057,221 | 3.05x | 22.2 | 54.4 % | 1.1 % | 0.0018 |
| `bf16-m1-k2560-n9216` | X8 + V3 | pass | 15,685,070 | 3.31x | 22.2 | 17.1 % | 0.0 % | 0.0018 |
| `bf16-m128-k2048-n6144` | gather (today) | pass | 445,561,279 | 1.00x | 37.5 | 8.9 % | 2.5 % | 0.0031 |
| `bf16-m128-k2048-n6144` | gather + V3 | pass | 446,983,865 | 1.00x | 37.5 | 2.6 % | 0.5 % | 0.0031 |
| `bf16-m128-k2048-n6144` | X8 | pass | 51,142,107 | 8.71x | 0.6 | 77.4 % | 0.3 % | 0.0031 |
| `bf16-m128-k2048-n6144` | X8 + V3 | pass | 55,522,804 | 8.02x | 0.6 | 20.6 % | 0.0 % | 0.0031 |
| `bf16-m128-k2560-n9216` | gather (today) | pass | 369,937,961 | 1.00x | 37.4 | 20.1 % | 3.4 % | 0.003 |
| `bf16-m128-k2560-n9216` | gather + V3 | pass | 366,658,288 | 1.01x | 37.4 | 5.8 % | 0.8 % | 0.003 |
| `bf16-m128-k2560-n9216` | X8 | pass | 97,086,186 | 3.81x | 0.4 | 76.5 % | 1.1 % | 0.003 |
| `bf16-m128-k2560-n9216` | X8 + V3 | pass | 95,285,005 | 3.88x | 0.4 | 22.5 % | 0.0 % | 0.003 |

"LSU instr / MMA" is `class_lsu / mma_commands` of the steady call. "MMA busy share" is the engine busy cycles over (3 cores × device cycles). "MMA wait share" is the share of core time stalled on the engine.

**Answer: V3 pays little even once the transpose is fixed.**

- **X8 itself:** FP8 linears go from 1.2–3.6× slower than BF16 to faster than BF16 at these shapes. For example m1-k2048: X8 10.7 M cycles against BF16 22.7 M. At m128 LSU work drops to 0.4–0.6 instructions per MMA, and the MMA busy share rises to 46–77 %.
- **V3 on top of X8:**
  - +9.7 % and +8.7 % at m1;
  - +1.9 % at m128-k2560, which is noise;
  - −8.6 % at m128-k2048.
- **Why:** the engine's wait share stays at 0.3–3.2 % even when it is busy 77 % of the time. Asynchronous MMAs overlap the rest of the work, so cutting busy time 3.5× buys at most about 10 %, and only for decode.

## Timing and area (unit OOC, x86 Vivado 2023.2, U280 part, 10 ns, `RCTX_W=2`)

OOC_RESULT

## Files

- `whatif_summary.py`: the what-if table, built from the run directories.
- `x8-evidence.tgz`, which holds:
  - what-if `summary.json` and the results/env/counters of every run;
  - fp8_cases reports (model quick and full, RTL quick, v5 quick);
  - `xpose_oracle.json`;
  - `validate_dma` validation.json for base and X8;
  - the regression logs;
  - the OOC reports.
