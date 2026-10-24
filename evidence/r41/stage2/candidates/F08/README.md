# F08: the FMA's middle stage (`p0_sticky_al` → `p1_sh_mag`)

**Status:** implemented on `r41s2-fma` (this commit, the last of F03 → F09 → F08), bit-exact, no cycle
change, no latency change. The middle stage's unit path projects to **8.70 ns** routed at 10 ns (target 9.0).
This is the final state of the series; its full regression is below. Not yet routed.

## The family

Attempt-3 routed census, residual population (`../F03/census_families.json`):

| Family (routed, 10 ns) | Endpoints | TNS (ns) | Worst slack | Its data path |
|---|---:|---:|---:|---|
| VFPU `…u_xls/p0_*` → `p1_*` | 95 | -32.9 | **-0.895** | 11.132 ns, 28 levels (13 CARRY8) |
| … of which `p0_sticky_al` → `p1_sh_mag` | 62 | -26.9 | -0.895 | 11.132 ns |
| MTX `…u_xls/p0_*` → `p1_*` | 45 | -10.0 | -0.638 | 10.526 ns |

The census path is the alignment's sticky bit → a second 81-bit carry chain (`sub_bs = (big − small_sh) −
sticky`, the first subtraction being in the first stage) → the magnitude select → the 89-bit leading-zero
count (one-hot and encode) → `msb = 88 − clz` → `e_norm = msb + exp_common − 78` → the subnormal compare →
`sh_extra = E_MIN − e_norm` → `sh_tot = sh_norm + sh_extra` → `|sh_tot|`: five serial carry chains after the
count.

## Design (`xls/tnca_fma.x`, `fma32_normalise`; the second cut)

- **Carry-select borrow.** `big − small_sh` and `big − small_sh − 1` (`big + !small_sh`) are formed side by
  side and the sticky bit only selects; `small_sh > big` is the sign of the first, so there is no separate
  80-bit comparator.
- **Everything after the leading-zero count is one compare or one subtraction from it.** With
  `lz = clz(mag)`, `msb = 88 − lz` and `e_norm = exp_common + 10 − lz`:
  - subnormal result ⟺ `e_norm < −126` ⟺ `lz > exp_common + 136` (the bound is formed beside the count);
  - the biased exponent before rounding is `(exp_common + 137) − lz`, or `E_MIN + 127` if subnormal;
  - the normalising shift of a normal result is `msb − 23 = 65 − lz` (left by `lz − 65` if `lz > 65`);
  - that of a subnormal result, `sh_norm + (E_MIN − e_norm)`, is `−71 − exp_common`: it does not depend on
    `lz` at all; its sign and magnitude are formed beside the count.
- **The second cut moves** to just after these: `FmaNormalised` (122 bits) carries both shifts and the
  subnormal flag, and the 2:1 select happens at the head of the last stage, which F09 left with slack.

### Why the result is unchanged

- `big + !small_sh = big − small_sh − 1 (mod 2^89)`; both operands are below 2^88, so bit 88 of
  `big − small_sh` is `small_sh > big`. When the sticky bit is set the result is positive (F03 README: the
  larger operand is at least 2^55, the shifted one below 2^47).
- The rewritten quantities are the old ones by substitution (above) whenever `mag != 0`. When `mag == 0`
  (`lz = 89`) they differ, but none reaches the result: the subnormal flag and the overflow test require
  `any`, and the result is the exact zero with the sign rule of the old code.

## Verification (GB10), final state

All on the final tree (this commit), GB10, each against r41s2 (`544aed8`) run the same way:

| Check | Result |
|---|---|
| DSLX unit tests (`interpreter_main`) | 2/2 pass |
| `tb_fp32_fma`, latency 2 and 3, exact Fraction oracle (`make unit`) | 41,240 vectors each, 0 failures |
| `make fma-deep`, latency 2 and 3 | 516,240 vectors each, 0 failures |
| Verilator miter against r41s2's generated RTL (`../F03/miter/`) | 700 M vectors × 2 latencies, 0 mismatches (this commit's logic) |
| `make unit` (fp32_fma, cvt_xls, fpu_multi, vector_fpu_f2i, sfu_xls/core, decoder, mul64, bitops) + `make core` + `make core-vrf` | 89 results, all PASS, **identical result lines to r41s2** (every `cycles=`) |
| `make core-slowmem core-ooo` | 78 PASS, identical to r41s2 |
| `make top r38-xpose-rtl` (2×2 system, mesh, transposed transfer) | 7 PASS, identical to r41s2 |
| `make core-r39sw5` (core-vrf-sync-v5: FPU_OVERLAP, MTX_ABUF, 4 RCTX, RPAIR; fast and out-of-order memory) | 84 PASS, identical to r41s2 |
| `rtl_oracle.py` (vector FPU: 71 ops — FMA family, min/max/cmp, all conversions, FP reductions, SFU — 28,400 vectors × 16 lanes) | `oracle.txt` identical to r41s2 |
| `matrix_oracle.py`, ABUF 1 and 0 (12 cases: four fragment dtypes × K16/32/64, row ops, m.zero, continuous commands) | `results.txt` (accumulator words and per-command busy cycles) identical to r41s2 |
| `tnca-numeric-probe matrix` (the SystemC model's independent matrix arithmetic) | 36,864 words per run, 0 mismatches, as r41s2 |
| `tnca-numeric-probe vfpu` | identical to r41s2: all FMA-family, reduction and SFU ops pass; the 2,460 mismatches are the documented f32→int conversion divergence (ops 18.2/18.3), the same set |
| `xlsgen.py --check` (FPGA host, pinned XLS) | every FMA module regenerates byte for byte |
| `make lint` | 70 files elaborate |
| Bitwise K-order (`fragment_case.py --bitwise [--cancel]`, sim-file, core-vrf-sync-v5), final and r41s2 side by side | running at this commit; results in the following evidence commit |

The MMA programs in these suites cover continuous MMA (`t11_matrix`, matrix oracle command streams),
accumulator RAW (MMA into an accumulator then `a.vread`/row ops on it; the engine's `acc_collide` check stays
armed), and the source-free / result-ready split (`t25_mmasf`: `src_free` before `done`).

## Cycle impact

None. FP_LAT stays 2: the VFPU result latency, the core's writeback reservation (`r_wb_delay`), the fold's
`FP_LAT + 1` cycles per level and the MTX's `PIPE = 3` are unchanged.

Representative programs (final = r41s2, every one):

| Program (profile) | cycles |
|---|---:|
| `t03_fp` (core / banked VRF / core-vrf-sync-v5 / v5 out-of-order memory) | 3,069 / 3,323 / 3,096 / 3,193 |
| `t11_matrix` (same) | 3,520 / 4,992 / 3,513 / 3,914 |
| `t25_mmasf` (same) | 1,290 / 1,889 / 1,242 / 1,510 |
| `t36_reduce` (same) | 8,323 / 8,923 / 8,307 / 8,453 |
| `t21_cvt` (same) | 1,942 / 2,068 / 1,898 / 2,887 |
| `t20_sfu` (same) | 1,282 / 1,354 / 1,248 / 1,388 |
| 2×2 system, `t10_mesh` all roles | 29,075 |
| matrix engine busy cycles per MMA command (ABUF 1): K16 FP16/BF16 287, FP8 283; K32 551/559; K64 1,095 | identical |

The bitwise K-order runs also record each Torch GEMM's device cycles on the same profile; they are compared
with r41s2's in the following evidence commit.

**SystemC model keys:** none changed. `vector_fp_latency = 5`, `convert_latency = 5`,
`fp_reduction_latency` (16 board / 12 three-core), `fp_reduction_vector_extra = 1`, `mtx_pipe_cycles`
(4 / 3), `mtx_row_cycles = 1` describe latencies the series does not alter (`tools/performance_harness/
systemc/configs/r40-*.cfg`).

## Unit OOC, 10 ns, xcu280-2L, image parameters

Method and scripts: `../F03/README.md`. Projection = after-synth × (census routed / baseline synth).

The middle stage has no inputs or outputs but the two cut registers, so its logic is the same in every
instance; the standalone `tnca_fp32_fma` (LAT 2) times it directly (in the census tree it is `p0_*` → `p1_*`,
in the final tree `s0_q` → `s1_q`).

| Unit path (family) | Baseline synth | Census routed | Ratio | After F03+F09 | Final | Projected (final) |
|---|---:|---:|---:|---:|---:|---:|
| middle stage, VFPU family (`p0_sticky_al` → `p1_sh_mag`) | 6.076 (35 levels) | 11.132 | 1.832 | queued | **4.747** (28 levels) | **8.70** |
| middle stage, MTX family | 6.076 | 10.526 | 1.732 | queued | 4.747 | 8.22 |
| middle stage, standalone `tnca_fp32_fma` | 6.230 (36 levels) | — | — | queued | 4.747 | — |

The whole series, final tree (`../F03/project.py`, `projection.json`):

| Family | Census worst (routed) | Unit path, synth: r41s2 → final | Projected routed |
|---|---:|---:|---:|
| F03 `red_busy` → first stage | -0.919 (10.879) | 6.760 → 5.371 | **8.64** |
| F08 middle stage (VFPU) | -0.895 (11.132) | 6.076 → 4.747 | **8.70** |
| F08 middle stage (MTX) | -0.638 (10.526) | 6.076 → 4.747 | 8.22 |
| F09 last stage → accumulator file | -0.871 (10.394) | 5.795 → 3.366 | **6.04** |
| F09 neighbour: last stage → VFPU reduction `acc` | -0.455 (10.218) | 6.084 → ≤ 5.0 | ≤ 8.40 |
| neighbour: MTX operand regs → first stage | -0.712 (10.767) | 5.514 → 4.856 | 9.48 |

Unit WNS at 10 ns: `tnca_vector_fpu` +3.184 → +3.976 ns (its worst path is now the converter, 6.006 ns,
outside the FMA), `tnca_matrix` +3.906 → +5.126 ns, `tnca_fp32_fma` +3.752 → +5.235 ns.

## Generated structure and FPGA mapping (the series)

What XLS generates, and what it maps to on the U280 (unit OOC, r41s2 → final):

| | r41s2 | final |
|---|---|---|
| latency-2 instance | `tnca_fma_xls`: `fma32` scheduled by XLS into 3 stages (asap7, minimum period) | `tnca_fma_{align,norm,round}_xls`: the three stage functions, combinational (opt IR 118 / 88 / 82 nodes); two cuts in `tnca_fp32_fma.v` |
| cut registers per FMA | XLS's `p0_*` / `p1_*` (351 FFs in the unit) | `FmaAligned` 150 + `FmaNormalised` 122 bits (271 FFs) |
| DSP48E2 per FMA | 2 | 2 |
| `tnca_fp32_fma` (LAT 2) | 2,332 LUTs, 351 FFs, 159 CARRY8 | 2,368 LUTs, 271 FFs, 145 CARRY8 |
| `tnca_vector_fpu` (16 FMAs, converters, SFU) | 62,755 LUTs, 9,677 FFs, 3,492 CARRY8, 46 DSPs | 62,286 LUTs, 8,393 FFs, 3,270 CARRY8, 46 DSPs |
| `tnca_matrix` (16 MACs, ABUF) | 50,092 LUTs, 12,774 FFs, 2,544 CARRY8, 32 DSPs | 50,744 LUTs, 11,493 FFs, 2,320 CARRY8, 32 DSPs |
| first stage, worst path | 27 levels (VFPU `red_busy` → `p0_res_sign_mag`) | 19 levels (`red_busy` → `s0_q`) |
| middle stage, worst path | 36 levels, 17 CARRY8 | 28 levels, 14 CARRY8 |
| last stage → accumulator file | 27 levels | 15 levels |
| latency-3 instance | `fma32`, XLS, 4 stages | the same function (the three composed), XLS, 4 stages |

Per tile (one VFPU, one MTX): +183 LUTs, −2,565 FFs; three tiles: +549 LUTs, −7,695 FFs. `xlsgen.py
--check` on the final tree: every FMA module regenerates byte for byte; the one stale file it reports,
`rtl/gen/tnca_widen_pair_xls.v`, predates this series (`xls/tnca_fpu.x` was edited in a9e3a1f without a
regeneration).

## Open risks

- Synthesis-only estimates scaled by the census's routed/synth ratio of the same path; only the routed
  kernel can confirm them. The kernel synthesis (`flow/fpga/synth_r40_multicore.tcl`) keeps the hierarchy
  (`-flatten_hierarchy none`); the stage modules are separate modules with only the cut registers between
  them, so nothing that the old single XLS module could optimise across a stage is lost except across the
  two cuts.
- The MTX's own first stage (registered operands → DSP → alignment shift) is not one of F03/F08/F09; it
  projects to 9.48 ns with its family's conservative census ratio (1.95; the routed path was -0.712 ns).
  Its remaining depth is the hidden-bit OR in front of the DSP and the shifter behind it.
- The VFPU's worst unit path is now outside the FMA: the float-to-float converter into the non-FMA delay
  line (`cvt_x` → `dly_v`, 6.006 ns synth, unchanged; census family 47 endpoints, -13.5 ns, -0.748 ns). It is
  not in this cohort.
- The VFPU → core `w_vdata` neighbour includes the core's writeback mux (core-owned).
- The latency-2 cuts are hand-placed. A deeper FMA for a faster clock is FP_LAT=3 (still XLS-scheduled from
  the same function) or a new split, not a config change.
- Cell names change (`g_l2.u_xls/p*_reg` → `g_l2.s0_q_reg`, `g_l2.s1_q_reg`, `g_l2.u_s0/1/2`); no
  constraint or census script in the tree matches the old ones (only a docstring example in
  `flow/synth/path_groups.py`). `tools/performance_harness/frequency_preflight.py`'s frozen Stage 1 owner
  list still names `rtl/gen/tnca_fma_xls.v`; that tool's source-identity check already refuses any changed
  RTL tree, so it is left as the Stage 1 record.

## Files

- `ooc_reports.tgz`: OOC reports of the final tree (no checkpoints).
- `regression_logs.tgz`: the regression and oracle logs above.
- `projection.json`: `../F03/project.py` over r41s2 and the three states.
