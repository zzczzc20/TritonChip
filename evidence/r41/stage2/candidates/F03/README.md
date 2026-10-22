# F03: VFPU `red_busy` into the FMA's first stage

**Status:** implemented on `r41s2-fma` (this commit), bit-exact, no cycle change, no latency change.
After the whole FMA series (F03, F09, F08) the family's unit path projects to **8.64 ns** routed at 10 ns,
under the 9.0 ns target. Not yet routed; the kernel build and the census are Stage 2 work.

## The family

From the attempt-3 routed census (`evidence/r40/v3/final_image_candidate/census/`, residual population;
`census_families.py` here, output `census_families.json`):

| Family (routed, 10 ns) | Endpoints | TNS (ns) | Worst slack | Its data path |
|---|---:|---:|---:|---|
| `u_vfpu/red_busy` → `g_fma[*]…u_xls/p0_*` (all) | 1,172 | -244.9 | **-0.919** (design WNS) | 10.879 ns, 23 levels |
| … → `p0_small_wins` | 142 | -54.7 | -0.919 | 10.879 ns |
| … → `p0_res_sign_mag` | 27 | -12.7 | -0.912 | 10.904 ns |
| … → `p0_sub_671` | 1,001 | -177.4 | -0.691 | 10.764 ns |
| neighbour: `x_va_q`/`x_vb_q` → VFPU `p0_*` | 308 | -63.9 | -0.813 | 10.387 ns |
| neighbour: `x_fpctl` → VFPU `p0_*` | 153 | -30.0 | -0.874 | 10.581 ns |

The worst path (census report, `residual_top_paths_full.rpt`) is `red_busy` (fanout 330) → LUT2
(`red_active = red_busy | fpctl[4]`, fanout 79) → the lane operand select → four LUTs of zero detection and
`ftz` decode (`a_zero` masking the significand) → two cascaded DSP48E2 (the 24×24 product) → the alignment
shifter → the 80-bit `small_sh > big` compare. 4.66 ns of the 10.88 ns are spent before the DSP's input, 7.4 ns
of the path is routing.

## Design

Three changes, none of which changes a result bit or a cycle.

1. **`rtl/core/tnca_vector_fpu.v`: the reduction tree's operands are selected on `red_busy` alone**, a
   flop, instead of `red_busy | fpctl[4]`. The fold presents its first tree level the cycle after `start`
   (red_busy is set together with the capture of `acc`); what the FMA array computes in the cycle the
   reduction itself sits in X is never used. That slot's FMA result is not retired (`out_from_fma` is false
   for major 0x15), and the sequencer reads `fma_y` only at `rwait == FP_LAT`, FP_LAT cycles after operands it
   presented itself. The OR and its ~900-load net were the head of the worst path. `fpctl[4]` is still driven
   by the core; the VFPU no longer reads it.

2. **`xls/tnca_fma.x`, first stage: no zero test in front of the exponent adders or the DSP.**
   - The exponent fields are used as stored (`x = e − 127`); a subnormal's fraction is doubled
     (`m = f << 1`) instead of its exponent being raised to 1. Every operand's value is the usual one.
   - The multiplier's operands are the raw fraction fields with the hidden bit (`e != 0`) and nothing else.
     A subnormal factor's doubling (a shift of the product by 0, 1 or 2) and the zero of an `ftz`-flushed
     factor are applied to the product after the DSP, where they fold into the alignment select.
   - `|xp − xc|` is two subtractions side by side instead of a negation behind one.

3. **The latency-2 FMA's two register cuts are placed for the FPGA** (`rtl/core/tnca_fp32_fma.v`,
   `tools/xls/config.json`). XLS balances its schedule against asap7, which prices the 24×24 multiplier —
   two DSP48E2 slices here — at several times its FPGA cost. Once the first stage was shortened, XLS's own
   three-stage schedule moved the alignment shifter into the middle stage (measured: `tools/xls` schedule
   probes at 1855–2400 ps; the shifter stays in stage 0 only from 2400 ps), which would make the middle stage
   the long one. `xls/tnca_fma.x` is therefore written as three stage functions, `fma32_align`,
   `fma32_normalise` and `fma32_round`, each generated combinational (`rtl/gen/tnca_fma_{align,norm,round}_xls.v`),
   and the wrapper registers their struct results `FmaAligned` (150 bits) and `FmaNormalised` at the two cuts,
   with the generated pipeline's synchronous reset. In this commit the second cut is where XLS had it (after
   the normalising shift amount); F08 moves it. `fma32`, the three composed, is the function every instance
   computes and what the latency-3 instance (`rtl/gen/tnca_fma_l3_xls.v`) is still scheduled from by XLS.
   `rtl/gen/tnca_fma_xls.v` (the XLS-scheduled latency-2 module) is removed; `Makefile` (`FMA_GEN`) and
   `flow/synth/synth.py` list the three stage modules instead.

The checked-in `xls/tnca_fma.x` before this commit did not parse with the pinned XLS (`u88`/`u89` are not
DSLX types; `uN[88]` is). Regenerating from it with `uN[…]` reproduced the checked-in `rtl/gen` byte for byte
apart from the recorded DSLX hash, so the old RTL was generated from an equivalent spelling. The new source
parses and regenerates (`tools/xls/xlsgen.py`, the pinned v0.0.0-10704-g0a7c502cc on the FPGA host).

### Why the result is unchanged

- *Raw exponents and doubled subnormal fractions.* A subnormal is `f·2^-149 = (2f)·2^(0−127−23)`. A zero's
  exponent reaches no result bit: a zero significand is never the reference operand (`shift_c`), and when
  both addends are zero the result is an exact zero. Everything after the unpack is value based — exact
  alignment with a sticky bit below a 32-bit guard field, an exact add, a leading-zero normalisation, one RNE
  rounding — so a value-identical representation gives the same result provided (i) nothing is lost at the
  top of the field (a subnormal fraction is below 2^23, so the doubled product still fits 48 bits and the
  fields stay below 2^80) and (ii) the sticky invariant still holds: a sticky bit means the alignment shift was
  at least 33, so `small_sh < 2^47`, while the larger operand is at least 2^55 (when the product is the larger,
  `ea + eb ≥ 127 + ec`, so a subnormal factor's partner is normal), so the normalising right shift is at least
  31 and the eps stays below the guard bit.
- *The `red_busy` select:* above (design, 1).
- *The cuts:* register placement does not change the function; two registers keep latency 2 and one result
  per cycle.

## Verification (GB10)

| Check | State | Result |
|---|---|---|
| DSLX unit tests (`interpreter_main`) | this commit, final | 2/2 pass |
| `tb_fp32_fma`, latency 2 and 3, exact Fraction oracle | this commit, final | 41,240 vectors each, 0 failures |
| `make fma-deep`, latency 2 and 3 | this commit, final | 516,240 vectors each, 0 failures |
| Verilator miter against r41s2's generated RTL (`miter/`), latency 2 and 3, 7 stimulus classes | this commit | 210 M vectors × 2 latencies, 0 mismatches |
| the same | final | 700 M vectors × 2 latencies, 0 mismatches |
| `rtl_oracle.py` (vector FPU, 71 ops incl. FP reductions, conversions, SFU; 28,400 vectors) | final vs r41s2 | identical `oracle.txt` (sha256 9aa4a5fe…) |
| `tnca-numeric-probe vfpu` on that oracle | final and r41s2 | identical: every FMA, reduction and SFU op passes; the 2,460 mismatches are the f32→int conversions (18.2/18.3) the model already documents, the same on both |

The miter (`miter/run_miter.sh`) drives both the candidate `tnca_fp32_fma` and the previous XLS-scheduled
modules every cycle with: all-random words; structured fields (zero/subnormal/extreme exponents, all-ones and
single-bit fractions); an exponent-difference sweep of ±100 around the product's; products cancelled by `c`
within ±4 ulps; subnormal inputs and results; all-ones significands for rounding carries and overflow; exact
short-significand products cancelled exactly. Control bits are random (`mul_only` 1/8, `sub_c` 1/2, `ftz` 1/4).
A mutation (dropping the alignment sticky from the round decision) is caught (96 mismatches in 1.4 M vectors).

The core and system regressions, the matrix oracle and the bitwise K-order check ran on the final state;
see `../F08/README.md`.

## Cycle impact

None. The FMA latency stays 2 (`FP_LAT`), the VFPU's and the MTX's latency tables and the core's writeback
reservation are untouched, and the reduction sequencer's timing is unchanged. Measured on the final state:
every core program's `cycles=` line is identical to r41s2 (`../F08/README.md`).

**SystemC model keys:** no change (`vector_fp_latency`, `convert_latency`, `fp_reduction_latency`,
`fp_reduction_vector_extra`, `mtx_pipe_cycles`, `mtx_row_cycles` describe latencies this does not alter).

## Unit OOC, 10 ns, xcu280-2L, image parameters

`run_ooc.sh` (FPGA host, through `~/r41s2_tools/vivado_slot.sh`): `flow/synth/vivado_ooc_timing.tcl` with
`OOC_SV=1`, then `fma_ooc.tcl`'s per-family report in the same session. Generics: `tnca_vector_fpu` at its
defaults (the core instantiates it so in the image), `tnca_matrix` with `ABUF=1` (`MTX_ABUF 1`),
`tnca_fp32_fma` with `LAT=2` (`FP_LAT` 2). Synthesis estimates, unplaced. Projection = after-synth ×
(census routed / baseline synth) of the family's path.

| Unit path (family) | Baseline synth | Census routed | Ratio | After F03 | After F03+F09+F08 | Projected routed (final) |
|---|---:|---:|---:|---:|---:|---:|
| VFPU `red_busy` → stage-0 regs (`p0_small_wins`) | 6.760 | 10.879 | 1.609 | see below | **5.371** | **8.64** |
| VFPU `red_busy` → `p0_res_sign_mag` | 6.798 | 10.904 | 1.604 | | 5.371 | 8.62 |
| VFPU `red_busy` → `p0_sub_671` | 6.556 | 10.764 | 1.642 | | 5.371 | 8.82 |
| MTX `g_in_reg.{a,b}_q` → stage-0 regs (neighbour, 99 endpoints, -0.712) | 5.514 | 10.767 | 1.953 | | 4.856 | 9.48 |

Baseline is r41s2 (`544aed8`); after is `rtl/gen` of the named state. In the final state every worst path into
the VFPU's stage-0 registers starts at `red_busy` (19 levels: the select, the hidden bit, the two DSPs, the
product select and the alignment shifter); the unit's worst path is no longer in the FMA at all but the
converter into the non-FMA delay line (`cvt_x` → `dly_v`, 6.006 ns, unchanged). Unit WNS: VFPU +3.184 →
+3.976 ns; MTX +3.906 → +5.126 ns.

*After F03 alone* (this commit's tree, second cut still after the normalising shift amount): its unit OOC is
queued on the shared FPGA host and is added by a later evidence commit; the column is filled there.

Area (final vs r41s2, OOC): VFPU 62,755 → 62,286 LUTs, 9,677 → 8,393 FFs, 46 DSPs both; MTX 50,092 →
50,744 LUTs, 12,774 → 11,493 FFs, 32 DSPs both.

## Open risks

- These are synthesis-only estimates scaled by the census's routed/synth ratio of the same path; routing in
  the final image is what decides. The x_va/x_fpctl neighbours are inputs of the VFPU, so they are not timed
  in its OOC; they share the stage-0 logic after the operand select and so its shortening.
- The MTX's own stage-0 path (registered operands → DSP → alignment shift) projects to 9.48 ns with the
  conservative census ratio of that family (1.95, a congested region). It was -0.712 ns in the census and is
  not one of F03/F08/F09; its remaining depth is the hidden-bit OR in front of the DSP (three LUTs) and the
  alignment shifter behind it.
- The latency-2 cuts are now hand-placed. A future clock that needs a deeper FMA is FP_LAT=3 (XLS-scheduled)
  or a new split, not a config change.

## Files

- `fma_ooc.tcl`, `run_ooc.sh`, `run_families.sh`: the OOC screen (also used by F08, F09).
- `miter/`: the Verilator miter (`fma_miter.v`, `fma_miter.cpp`, `run_miter.sh`).
- `census_families.py`, `census_families.json`: the census families above.
- `ooc_reports.tgz`: OOC summary, path, CSV, family and utilisation reports (no checkpoints) of r41s2 and of
  this commit's tree.
