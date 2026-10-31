# R41 Stage 2 integration (S2-00, S2-01): merge, regression and pre-link screens

Branch `r41s2-int`. No v++ and no full link was run. The U280 card and the FPGA host's queues were not touched.

**Recommendation: GO for the one integration link, as a measurement run. Do not expect 100 MHz from it.**
The screens predict a routed worst slack of about −0.55 to −0.9 ns. Most of it comes from three tile-internal
families (see "Expected to fail"). They also leave the families outside a tile unmeasured.

## Merge record

| Step | Commit | Content | Resolution |
|---|---|---|---|
| base | `708a920` (`r41s2`) | F07/C1 and the base screen evidence | — |
| merge `r41s2-fma` | `390d258` | F03, F09, F08 (FMA stages, XLS regen) | clean |
| merge `r41s2-core` | `c11a78c` | F01, F02, F04, F05 and the WIP `2928c85` | clean. The WIP was **kept and finished**, see below |
| merge `r41s2-io` | `cf28fe3` | F06, and F10 as WIP `46b72a8` | `Makefile`: kept both new targets (`r41-core-boundary` and `lsu-reqreg`). The F10 WIP was **kept**, see below |
| tooling and regression | `b837c3c` | `regional_preflight.py --kernel-synth`, `screen_families.py`, the cf28fe3 regression logs | — |
| merge `r41s2-x8` | `8aca2a9` | X8, the 1-byte transposed write (`HAS_TENSOR_XPOSE=2`), by coordinator decision | clean |
| image parameters | `45b37bc` | `image.tcl` / `image.json`: the attempt-3 parameters with `HAS_TENSOR_XPOSE 1 → 2`, profile `core-vrf-sync-v5-x8` | — |

The packed FP8 MAC (V3, `r41-pmac`) is **not** in this tree or this image.

**Core WIP `2928c85`.** It held three things:
- **F02, completed.** The transfer and MMA scoreboard indices `Sb+1..+4` and the `r_mma_sf`/`r_txfer` opcode tests are decoded in D and registered with R, under R's own `!stall_front` enable.
- **F04, completed.** The writeback data is built as one AND-OR over one-hot selects, replacing the nested priority chains.
- **F01.** `max_fanout 200` on `stall_front`.

The WIP was functionally complete, and both bench assertions it adds held in every core run (no `FAIL tnca_core assert` anywhere):
- R's registered indices equal what R would compute itself;
- the AND-OR writeback equals the priority chains.

Every core target without `LSU_INFLIGHT=2` matches the r41s2 base cycle for cycle. The candidate READMEs F02/F04 describe only the earlier parts; this README is the record for the WIP part.

**IO WIP `46b72a8` (F10, registered LSU request).** It is complete in RTL and has its functional test: `make lsu-reqreg` is a miter against `REQ_REG=0`, with 5 runs, back-pressure, reordering, and a directed latency check. All runs pass here, as does the full integrated regression.

Its README placeholders (`RESULT_DMA`, `RESULT_NATIVE`, `NATIVE_ROW`) are now filled with what was actually measured. The per-candidate archives it cites (`f10-regression-logs.tgz`, `f10-ooc-evidence.tgz`) were never committed, and the F10 README now says so.

## Functional regression (GB10, iverilog 12, Verilator 5.020)

Logs: `regression-cf28fe3.tgz` (lanes A–J; base runs on an `r41s2` archive) and `regression-x8-8aca2a9.tgz`. Cycle comparisons go case by case against the r41s2 base. The core base logs are the Stage 2 core agent's `r41s2` runs. The multicore and pipeline base runs were repeated here on `git archive origin/r41s2`.

| Target | Result | Cycles vs r41s2 |
|---|---|---|
| `core`, `core-slowmem`, `core-ooo`, `core-vrf` | 40 PASS each | identical (39 common cases; `t41_boundary` is new) |
| `core-r39hw` | 80 PASS | identical |
| `core-r39sw` / `core-r39sw4` / `core-r39sw5` | 80 / 80 / 86 PASS | **+0.79 % / +0.69 % / +0.64 %** in total; worst case `t18_squash` +8.7 % |
| `r41-core-boundary` (45 runs, every bench × 1/2/all contexts) | 45 PASS | +0.80 % against the core-only final tree (F10) |
| `top`, `r38-xpose-rtl` | 4 + 3 PASS | identical |
| `lsu-reqreg` (F10 miter) | 5 PASS | REQ_REG=1 costs +1 cycle per access (directed), stream 127,874 → 128,404 |
| `unit`, `fma-deep` (L2/L3, 516,240 vectors each), `lint` | PASS | — |
| FMA miter (`../candidates/F03/miter`) against r41s2's generated FMA | 210 M vectors × 2 latencies, **0 mismatches** | — |
| `rtl_oracle.py` (vector FPU, 400/op) and `matrix_oracle.py` (12 cases, ABUF 1 and 0) | outputs **byte-identical** to r41s2 (`oracle.txt` 9aa4a5fe…, results e0c261d0… / 70b6dd1f…) | MTX busy cycles identical |
| `r40-multicore-shell` | 20 PASS | grids +1.13 % |
| `r40-line-axi` (N = 1–8, deadline, fault drain) | 101 PASS | grids +0.89 % |
| `r40-grid-dispatch` | 359 PASS | +4 cycles per grid (F06 admission), +0.01 % |
| `r40-cache-maintenance`, `r40-multicore-dma`, `r40-coherent-host`, `multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top` | PASS | — |
| `compiler-rtl`, `triton-e2e`, `r40-barrier-epoch`, `r40-core-parameters` | PASS | — |
| `r40-pipeline-checks` (MMA/LSU/barrier overlap, `rctx_roles` at 2 and 3 stages, native pipeline) | PASS. The first try (F.04) failed on the missing `torch` in `TRITON_PYTHON`, as the base did; rerun with the torch env for both trees (H, I) | 16 runs: **+1.55 %** (overlap off) / **+1.32 %** (on); range −0.4 … +3.1 % |
| **After the X8 merge:** `r41-xpose8-rtl` | 7 PASS | — |
| X8: `core`, `core-r39sw5`, `top`, `r38-xpose-rtl` | PASS | identical to cf28fe3 |
| X8: `core-r39sw5` elaborated with `HAS_TENSOR_XPOSE=2` (the image's core) | 86 PASS | identical to cf28fe3 |
| X8: `unit`, `compiler-rtl`, `triton-e2e` | PASS | — |

**Cycle changes.**
- **F10 (`REQ_REG=1`).** Every `LSU_INFLIGHT=2` configuration pays +1 cycle per LSU access before the first request. This explains the +0.6–0.8 % on the v5-family core benches, about +1 % on multicore grids and +1.3–1.6 % on compiled pipeline kernels.
- **F01.** Only `t39_switch` changes (−52/−88 cycles on v4/v5).
- **F06.** +4 cycles per grid admission.
- **No change** from F02, F04, F05, F03/F08/F09 (bit-identical FMA and MTX timing) or X8 (v5 cycles unchanged).
- **Semantics.** Unchanged everywhere a comparison exists.

Not run by this regression:
- the native Torch rows of F10;
- the 113-point default-change audit. X8 is capability-gated, and its own README reports the target files byte-identical.

## Physical screens (x86 Vivado host, Vivado 2023.2, xcu280-fsvh2892-2L-e, 10 ns)

Raw reports: `screens-evidence.tgz`. It holds the timing and utilization reports, the failing endpoints, the 2,000 worst paths per screen as TSV, the scripts and the run record. The checkpoints stay on the host under `~/r41s2int`.
- **Kernel synthesis:** `flow/fpga/synth_r40_multicore.tcl` with the image parameters, at 6 threads.
- **Tile screens:** `flow/fpga/screen_r41_tile.tcl` at 4 threads, at most 2 jobs at once.
- **"x1" and "x8":** the `cf28fe3` tree with attempt-3 parameters (`HAS_TENSOR_XPOSE=1`), and the `8aca2a9` tree with `image.tcl` (`HAS_TENSOR_XPOSE=2`). **x8 is the image.**
- **Host setup** (no sudo). Vivado was copied to `~/Xilinx`, and `settings64.sh` was repointed from `/tools/Xilinx` (the original is kept as `.orig`). Three missing pieces were supplied in user space:
  - `libtinfo.so.5`: Ubuntu 22.04's 6.3-2ubuntu0.3, copied to `~/Xilinx/compat-lib` and put on `LD_LIBRARY_PATH`;
  - the `en_US.UTF-8` locale, which `rdiArgs.sh` forces: built with `localedef` into `~/.locale` (`LOCPATH`);
  - an empty `SharedData/2023.2/data/parts/xilinx/devint/vault`, which only holds Versal data.

  A routed xcu280 smoke design passed (WNS +9.344).

### Kernel synthesis (unplaced)

| Tree | WNS | TNS | LUTs | FFs | Worst path |
|---|---:|---:|---:|---:|---|
| r41s2 base 544aed8 (FPGA host) | +1.740 | 0 | 810,905 | 241,891 | — |
| int x1 cf28fe3 | **+4.008** | 0 | 807,339 | 231,221 | tile 0 VFPU `cvt_x` → `dly_v` |
| **int x8 8aca2a9 (image)** | **+4.008** | 0 | 810,980 | 231,282 | same |

Per tile (x8 against base): +751 / +157 / −627 LUTs and about −3,600 FFs each. The FF drop is F08's −2,565 FFs per tile plus the registered boundaries.

**SLR estimate** (`regional_preflight.py --kernel-synth`: the A3 routed counts, plus the M1–M4/F07 unit deltas, plus this kernel delta, with each tile charged to its own SLR and the rest of the kernel charged to SLR0):

| SLR | A3 routed LUT % / CLB packing | int x1 estimate | **int x8 estimate** |
|---|---|---|---|
| SLR0 | 73.45 % / 95.24 % | 73.32 % / ~95.08 % | **73.61 % / ~95.44 %** |
| SLR1 | 75.24 % / 88.39 % | 75.12 % / ~88.25 % | 75.48 % / ~88.67 % |
| SLR2 | 65.21 % / 85.41 % | 65.05 % / ~85.20 % | 65.25 % / ~85.46 % |

SLR0 packing stays at A3's level, about 95.4 %. Registers fall by about 2,700 per SLR, but LUTs, which set packing, do not. The packing figure is a linear scale, not a placed result.

### Tile place-and-route screens (one tile alone in one SLR, routed)

| Screen | Placed WNS | **Routed WNS** | TNS | Fail | WHS | LUTs | FFs | CLBs | Endpoints < +0.5 / < +1.0 ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base tile 1, SLR1 (FPGA host, 2026-10-21) | +0.530 | +0.015 | 0 | 0 | +0.019 | 249,646 | 68,399 | 41,236 | 544 / ≥2,000 |
| base tile 0, SLR0 (calibration, new) | +0.502 | +0.102 | 0 | 0 | +0.006 | 248,037 | 68,432 | 40,853 | ≥2,000 / ≥2,000 |
| int x1 tile 1, SLR1 | +0.576 | +0.219 | 0 | 0 | +0.007 | 245,175 | 64,720 | 41,550 | 29 / 280 |
| int x1 tile 0, SLR0 | +0.804 | +0.414 | 0 | 0 | +0.007 | 245,377 | 64,859 | 41,358 | 2 / 61 |
| **int x8 tile 1, SLR1** | +0.851 | **+0.261** | 0 | 0 | +0.015 | 246,761 | 64,841 | 41,758 | 178 / ≥2,000 |
| **int x8 tile 0, SLR0** | +0.750 | **+0.595** | 0 | 0 | +0.007 | 246,637 | 64,868 | 40,825 | 0 / 45 |

All screens show 39.5 BRAM tiles and 171 DSPs. "≥2,000" means the 2,000-path report was exhausted.

**Families by owner, margin +1.0 ns** (`screen_families.py`, `screen_families_m1.0.json`; `_m0.5` also included). Families use the A3 census keys.

| Owner | base t1 | base t0 | x8 t1 | x8 t0 |
|---|---|---|---|---|
| F04 writeback (`x_pm_idx`/`x_func`/`of_d1` → `w_sdata`/`w_vdata`) | 274 ep, +0.015 | 2 ep, +0.353 | 50 ep, +0.378 | 21 ep, +0.595 |
| F03/F08 VFPU/FMA (`red_busy` → FMA p0) | 279 ep, +0.222 | 82 ep, +0.102 | — | — |
| W forward into operands and VRF banks (`w_idx`/`w_class` → `x_v*_q`, LSU lane regs, bank `s1_vld`; F01/F02/F05 residue) | 53 ep, +0.157 | 1,916 ep, +0.321 | 197 ep, **+0.261** | — |
| Scoreboard/VRF (`u_srf/valid` → `sb_s`, bank state) | 464 ep, +0.193 | — | — | 2 ep, +0.745 |
| **Transfer engine** (`u_txf/t_e4`/`t_c1` → `xbuf`; not in F01–F10) | 646 ep, +0.375 | — | **1,723 ep, +0.291** | 22 ep, +0.897 |
| F09 MTX operand capture | 64 ep, +0.285 | — | 30 ep, +0.674 | — |

**What moved.**
- **Gone from the near-critical set:** the F03/F08 FMA family, A3's largest at 1,001 endpoints and −0.691 ns.
- **Improved:** the F04 writeback, +0.015 → +0.378/+0.595.
- **Improved on the integrated tree:** the scoreboard families.

Worst tile slack improves by +0.25 ns (SLR1) and +0.49 ns (SLR0) over the base. The x1 and x8 screens differ by run-to-run placement as much as by X8: tile 1 gives +0.219 versus +0.261, with different families on top.

## Calibration and prediction for the linked image

These screens are optimistic. One tile sits alone in an SLR, with no shell, no other tiles, no SLR crossing and none of SLR0's static density.

**Calibration.** It uses every A3 census family with at least 5 endpoints that also appears in a base screen: 15 families in tile 1 and 5 in tile 0. Full-image slack minus screen slack ranges from **−0.68 to −1.33 ns, median about −1.07 ns**. This calibration has known limits:
- A3 predates M1–M4 and C1, so the pairing is by family name, not by identical RTL.
- No routed full image of r41s2 exists.

**Prediction.** Applying −0.8 … −1.15 ns to the image screens (x8) gives:
- tile 1: worst about **−0.55 … −0.9 ns**, with 2,000 or more endpoints within 1 ns of the screen's edge;
- tile 0: worst about **−0.2 … −0.55 ns**, 45 endpoints within 1 ns.

That is roughly A3's −1.0 ns improved by 0.1–0.45 ns, with a much smaller tile-0 population. **100 MHz is not predicted to close on the first link.**

### Expected to fail at 100 MHz (tile-internal, by screen evidence)

1. **Transfer engine `u_txf` → `xbuf` (`t_c1`/`t_e4`, 19–22 levels).** It is not owned by any of F01–F10. A3 already had 298 failing `u_txf` endpoints (worst −0.494). In the x8 tile-1 screen it is the largest near-critical population: 1,672 endpoints in one family, at +0.291. X8 edits this unit (`x_data` byte select). The family exists without X8 too (base: 496 endpoints at +0.375).
2. **W forward/fan-out into operand capture and VRF bank state** (`w_class`/`w_idx` → bank `s1_vld`/`s1_byp_mask`, `x_v*_q`, LSU lane registers). This is the remainder of the A3 `w_en` → operand families after F01/F02/F05: +0.261 in x8 tile 1, with about 8.5 ns of route on 10–19 levels.
3. **F04 writeback remainder** (`of_d1`/`x_func` → `w_vdata`, 15–26 levels): +0.378 / +0.595.

### Not covered by any screen (risk families of the A3 census outside a tile)

- F10's second family and the residual quiescence cone (LSU FSM → `core_idle` → `&tile_idle` → handoff control, MMIO `control`), which crosses SLRs;
- F06 grid-dispatch admission;
- the mesh router FIFOs (F10's first family now ends inside the LSU queue: `t_i0` → `q_h*` at +0.278 in the x1 tile-1 screen);
- every SLR-crossing and shell path;
- SLR0 packing at about 95.4 %.

The kernel synthesis gives these only an unplaced slack (+4.0 ns), which predicts nothing for routing.

## Go / no-go

**GO for the single integration link, as the routed calibration point. Do not count it as a 100 MHz attempt.**
- Every candidate is functionally clean. Cycle costs are small and accounted for: F10 is +0.6–1.6 %, and the rest are zero or a few cycles.
- The screens show real tile-internal gains: the FMA family is removed, and F04, F05 and F01 move by +0.25 to +0.5 ns. The kernel's unplaced WNS goes from +1.74 to +4.01.
- No screen can price the outside-tile families or SLR0 density. The corrective implementation needs a routed census of this tree, which only the link can give.

**If the lead wants a chance at closure on this link instead,** the transfer-engine `xbuf` family (item 1) is the one to fix first. It has the largest population, no F-candidate owns it, and X8 touches it. The W-forward fan-out (item 2) comes second. Either fix needs its own functional and screen pass before the link.
