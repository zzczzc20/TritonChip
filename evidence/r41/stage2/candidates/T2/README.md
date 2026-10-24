# R41 Stage 2 T2: transfer-engine and W-forward timing families

Branch `r41s2-t2`, from `r41s2-int` at `e843bc4`. No full link was run, and neither the U280 card nor the FPGA host was touched.

**Result.** Families 1 and 2 are closed in both tile screens: no endpoint is below +1.5 ns in round 3. Family 3 (F04 writeback) is not fixed, and it now sets the worst slack of both tiles.

## Commits

| Commit | Fix | Content |
|---|---|---|
| `8a333a0` (WIP checkpoint, both fixes, round 1) | 1 | Per-state handshakes in `tnca_tensor` (`dl/ld/st/x_fire`). `tnca_mem_mux`: `b_req_ready = m_req_ready`; `pick_b` is `b_req_valid`, so the fire is unchanged. The `xbuf` zero write no longer depends on the handshake. The start-address product is registered (`mul_q`, `t_pv`). |
| | 2 | `d_whit`/`r_whit`: `WB_HIT_V` for D's and R's three vector sources, computed a cycle early from `nw_*` and D/R's next contents. `of_acc_q`: the response-arrival bits are the previous cycle's registered accept. |
| `1332058` | 1 | `t_v` holds the per-position row bounds as registers. `t_ps` = `t_P3 + t_CB`, so `r_m` and the request addresses are one add from registers (they were four). `x_oob_q` is the registered unit bounds of the transposed walk. `x_q` advances on every `S_XWRITE` cycle, and `x_held` undoes the advance when the write is held, so `x_q` no longer waits for `vw_ready`. |
| `89a6441` | 2 | `txf_vw_ready` comes from W's bank, registered with `w_*` (`w_vec_q`, `w_bank_q`), instead of from the file's write arbitration. The prefetch port address select `pf_sel` no longer depends on the hazards, which now gate only the valids. X's extra scalar read indices (`x_sd_q`..`x_sg_q`) are registered. |

**Equivalence.** Every new register is asserted every cycle against the logic it replaces. The assertions print `FAIL tnca_core/tnca_tensor assert`, which `run_test.sh` treats as a failure. No assertion fired in any run.

**Cycle change.** The only change is the registered product: one more `S_TMUL` cycle per transfer of rank ≥ 2.

## Regression on the final tree `89a6441` (GB10)

Logs are in `regression-t3-89a6441.tgz`. Cycles are compared case by case with the integration tree's logs (`r41s2-int-logs`).

| Target | Result | Cycles vs r41s2-int |
|---|---|---|
| `core` / `core-slowmem` / `core-ooo` / `core-vrf` | 40 PASS each | +0.030 / +0.007 / +0.014 / +0.024 %; only transfer tests change (`t12_tensor` +4, `t33_tdesc_load` +12) |
| `core-r39hw` / `core-r39sw` / `core-r39sw4` / `core-r39sw5` | 80 / 80 / 80 / 86 PASS | +0.027 / +0.052 / +0.050 / +0.059 % |
| `core-r39sw5` with `HAS_TENSOR_XPOSE=2` (the image's core) | 86 PASS | +0.059 % |
| `r41-core-boundary` | 45 PASS | identical |
| `r41-xpose8-rtl`, `r38-xpose-rtl`, `top`, `lsu-reqreg`, `unit`, `fma-deep`, `lint` | PASS | — |
| `compiler-rtl`, `triton-e2e` | PASS | — |
| `r40-multicore-shell` (20), `r40-line-axi` (101), `r40-grid-dispatch` (359) | PASS | simulated end times identical |
| `r40-cache-maintenance`, `r40-multicore-dma`, `r40-coherent-host`, `r40-barrier-epoch`, `r40-core-parameters`, `multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top` | PASS | — |
| `r40-pipeline-checks` (16 kernels) | PASS | +0.05 % with overlap off, +0.20 % with overlap on; worst kernel +0.22 % |
| FMA miter against the base's generated FMA | 210 M vectors × 2 latencies, 0 mismatches | Run on the round-2 tree. T2 touches only `tnca_core`, `tnca_tensor` and `tnca_mem_mux`, so the FMA is unchanged |

## Screens (x86 Vivado 2023.2, image parameters, 10 ns)

Reports are in `screens-t2.tgz`. It also holds `near_endpoints_1p5.tsv`, which lists every endpoint below +1.5 ns. The `screen_families.py` output for round 3 is `screen_families_t2_m1.0.json`.

**Kernel synthesis.** WNS +4.008 ns, the same VFPU path as e843bc4. LUTs 810,980 → 810,937. DSPs per tile stay at 171.

Cells give the worst slack (ns) / the number of endpoints below +1.4 ns. For e843bc4 the only data is the 2,000-path report.

**Tile 1 (SLR1)**

| Family | e843bc4 | r1 `8a333a0` | r2 `1332058` | r3 `89a6441` |
|---|---|---|---|---|
| Transfer engine, internal (`u_txf` → `u_txf`) | +0.291 / 1723 | +0.668 / 2376 | +1.391 / 1 | ≥ +1.5 |
| …of which walk → `xbuf` | +0.291 / 1672 | +0.668 / 1557 | ≥ +1.5 | ≥ +1.5 |
| W forward (`w_idx/en/class/rctx` → any) | +0.261 / 197 | +0.443 / 20 | +0.892 / 154 | ≥ +1.5 |
| VRF bank read side (→ bank `s1_*`/mem) | +0.261 / 197 | +0.760 / 66 | ≥ +1.5 | +1.186 / 14 |
| NI response → `u_txf` (`rsp_tag` → `xbuf`, 2 levels) | > +0.99 | +1.349 / 1 | +1.130 / 34 | +0.977 / 26 |
| F04 writeback (→ `w_vdata`/`w_sdata`) | +0.378 / 50 | +0.300 / 221 | +0.756 / 58 | **+0.507 / 180** |
| Tile worst | +0.261 | +0.300 | +0.756 | **+0.507** |

**Tile 0 (SLR0)**

| Family | e843bc4 | r1 | r2 | r3 |
|---|---|---|---|---|
| Transfer engine, internal | +0.897 / 965 | ≥ +1.5 | ≥ +1.5 | ≥ +1.5 |
| W forward | +1.376 / 1 | +1.202 / 4 | +1.445 / 0 | ≥ +1.5 |
| VRF bank read side | +0.745 / 39 | +1.046 / 6 | +0.944 / 98 | +1.100 / 9 |
| NI response → `u_txf` (`rsp_rdata` → `dl_buf`, 0 levels) | +1.264 / 2 | +0.980 / 10 | +1.415 / 0 | +0.285 / 18 |
| F04 writeback | +0.595 / 134 | +0.329 / 127 | +0.598 / 135 | **+0.159 / 220** |
| Tile worst | +0.595 | +0.329 | +0.362 | **+0.159** |

What remains:
- **F04.** The worst path is `of_d1` → the vector ALU's DSP multiply and MULH correction (25 levels), then the lane mux, `vres` and `w_vdata`. The other F04 paths are `x_func`/`x_imm` → the integer reduction (min/max tree), broadcast into `vres`. Their slack swings by 0.6 ns between placements (+0.16 to +0.76).
- **Transfer-engine families left.** The NI response captures into `xbuf`/`dl_buf`, with 0–2 logic levels. These are pure placement: they ranged from +0.29 to +1.42 across runs.
- **Bank read side.** The arbitration paths from `r_*`, `u_mtx` and `of_acc_q` to bank `s1_vld` stay at +0.94 to +1.19 ns.

## Projection for the linked image

This applies the integration README's calibration (image minus screen = −0.68 … −1.33 ns, using −0.8 … −1.15) to round 3:
- **Families 1 and 2:** ≥ +1.5 in the screen, so about +0.35 … +0.7 ns in the image. They are no longer limiters.
- **Tile 1:** worst about **−0.3 … −0.65 ns** (F04).
- **Tile 0:** worst about **−0.65 … −1.0 ns** (F04, on a worse placement).

Across the four placements, F04 alone predicts −0.05 … −1.0 ns. The families outside the tiles are still unmeasured, as before.

## Go / no-go for full run #1

**No-go as a 100 MHz attempt.** F04 is predicted to fail by 0.3–1.0 ns. **Go only as a measurement run**, if the lead wants a routed census of this tree: it is functionally clean, and the costs are +0.01–0.06 % on core benches and +0.05–0.2 % on pipeline kernels.

**Recommended next fix: F04.** Take the vector integer multiply/MULH and the reduce broadcast out of the single-cycle X stage. They would go to the scheduled-writeback path that the scalar multiply already uses (`MUL_LAT`, `wb_rsv`). That is an extra cycle of latency for those operations only.

## Round 4: F04 and the NI response captures (`2549feb`)

The coordinator asked for both fixes after `c9f2578`.

**F04.** `tnca_vector_alu` has a new parameter, `LATE = 1`. With it, the multiply family (`v.mul`, `v.muli`, `v.mulh.s`, `v.mulh.u`) and the integer reductions are taken off `vres`/`sres` and get their own outputs, `late_v`/`late_s`.
- **Write-back.** The core registers them in `vl_*` and writes them back one cycle after X. They are booked like the scalar multiply (`r_wb_delay` 1, `wb_rsv`), so the scoreboard, forwarding and writeback arbitration (`ws_vl`) handle them like any delayed result. These operations gain +1 cycle of result latency; no other operation changes.
- **Equivalence.** A bench copy of the unit with `LATE = 0` (`u_valu_ref`) runs on X's operands. Its result, registered, must equal every late write-back: `FAIL tnca_core assert`. It held in every run.

**NI response captures.** `dl_buf` and `xbuf` are now written one cycle after the response, from a registered copy (`rq_*`). Nothing reads them that early:
- `S_DPARSE` and the `dl_boxel` fold read only words 0–13;
- `S_XWRITE` starts two cycles after the last response.

There is no cycle change.

### Regression on `2549feb`

`regression-t4-2549feb.tgz`. Everything passes, and no assertion fired.

| Target | Result | Cycles vs r41s2-int |
|---|---|---|
| `core` / `core-slowmem` / `core-ooo` / `core-vrf` | 40 PASS each | +0.52 / +0.40 / +0.47 / +0.51 % |
| `core-r39hw` / `core-r39sw` / `core-r39sw4` / `core-r39sw5` | 80 / 80 / 80 / 86 PASS | +0.49 / +0.51 / +0.50 / +0.49 % |
| `core-r39sw5` with `HAS_TENSOR_XPOSE=2` | 86 PASS | +0.49 % |
| `r41-core-boundary` | 45 PASS | +0.05 % |
| `r41-xpose8-rtl`, `r38-xpose-rtl`, `top`, `lsu-reqreg`, `unit`, `fma-deep`, `lint`, `compiler-rtl`, `triton-e2e` | PASS | — |
| `r40-multicore-shell` (20), `r40-grid-dispatch` (359), `r40-line-axi` (101) | PASS | simulated end times identical |
| The other r40 suites and `multicore_mmio_tests` | PASS | — |
| `r40-pipeline-checks` (16 kernels) | PASS | +0.34 % with overlap off, +0.65 % with overlap on; per kernel +0.01 to +2.2 % |
| FMA miter | 210 M × 2, 0 mismatches | — |

Most of the core-bench cost is in two tests:
- `t36_reduce`: 8,323 → 8,811 cycles, +5.9 %, back-to-back dependent reductions;
- `t01_vector`: 943 → 962 cycles, +2.0 %.

### Screens (round 4)

**Kernel synthesis.** WNS is +4.008 ns, unchanged. LUTs go from 810,980 to 814,188, about +1,070 per tile. Routed tile LUTs: 246,637 → 247,714 (tile 0).

| Family (worst ns / endpoints < +1.4) | Tile 1, r3 | **Tile 1, r4** | Tile 0, r3 | **Tile 0, r4** |
|---|---|---|---|---|
| F04 writeback (→ `w_vdata`/`w_sdata`/`w_pdata`) | +0.507 / 180 | **≥ +1.5** | +0.159 / 220 | **+1.431 / 0** |
| The moved logic, now ending at `vl_vdata` | — | +1.124 / 12 | — | **+0.885 / 130** (`w_pdata` → predicate mask → reduction tree) |
| NI response → `u_txf` | +0.977 / 26 | ≥ +1.5 | +0.285 / 18 | ≥ +1.5 |
| Transfer engine, walk → `xbuf` | ≥ +1.5 | ≥ +1.5 | ≥ +1.5 | ≥ +1.5 |
| Transfer engine, other internal | ≥ +1.5 | +1.193 / 33 (load-slot completion → `vw_data`) | ≥ +1.5 | ≥ +1.5 |
| W forward | ≥ +1.5 | +1.320 / 1 | ≥ +1.5 | ≥ +1.5 |
| VRF bank read side | +1.186 | +1.500 / 0 | +1.100 | ≥ +1.5 |
| **Tile worst** | +0.507 | **+1.054** (F09 MTX) | +0.159 | **+0.690** (F03/F08 VFPU `of_d0` → `dly_v`) |

In round 4 two of the families closed earlier came back at +1.19 to +1.32 ns in tile 1, with 1–33 endpoints:
- the transfer engine's load-slot completion path into `vw_data`;
- one W-forward endpoint.

These logic cones did not change in round 4, so the movement is placement.

Families below +1.4 ns in round 4 that this work does not own:
- **F03/F08 VFPU:** +0.69 ns (tile 0), +1.16 ns (tile 1);
- **F09 MTX operand capture** (`u_mtx/st` → `g_mac` input registers): +0.81 ns (tile 0), +1.05 ns (tile 1);
- **`x_rctx` → `sb_s`:** +1.09 ns (tile 0);
- **F07 slice LUTRAM addresses:** +1.04 ns (tile 0).

**Projection** (image = screen −0.8 … −1.15 ns):
- tile 1: about +0.25 … −0.1 ns;
- tile 0: about −0.1 … −0.46 ns. This is set by the VFPU, and by the new `vl_vdata` stage at +0.885 ns.

The families outside the tiles are measured by full run #1 (`c9f2578`).

**Next step to make F04 robust**, if it is wanted: split the late stage. Register the reduction's masked inputs and the DSP products in X, then finish the tree, the MULH correction and the broadcast in the `vl` stage. That moves about 0.5 ns off `vl_vdata` without adding latency.

## Rounds 5 and 6 (`9f22689`, `3643bfb`)

**Round 5, `9f22689`.**
- **The late stage is split.** `tnca_vector_alu` `LATE = 2`: in X the unit registers the products, the MULH correction, the reductions' four-lane partial sums and bitwise partials, and the min/max keys with their pairwise compares. The cycle after X finishes them, so latency is unchanged from round 4.
- **VFPU.** The worst path was `of_d0` → the int↔f32 `vf.cvt` (lane 15, 21 levels) → `dly_v[0]`. That conversion now joins the delay line at stage 1, from the registered stage-0 inputs. The `vf.cvt.fmt` conversion already did this. The cycle and the result are the same, and a bench assertion compares every lane with the stage-0 value.
- **Matrix.** `S_ROW` and its row op become flags registered together with the command (before: `st` → `fmt` → MAC input registers). The `a.vwrite` payload is copied every cycle; before, the `r_rowop` decode drove the enable of every payload bit.

**Round 6, `3643bfb`.** The vector FPU gets its own copy of X's operands (`keep`), so the copy can be placed next to the FMA lanes. Its value equals `x_v*_q` every cycle, which is asserted.

### Regression (logs in `regression-t5-t6.tgz`)

Both trees pass everything in the list, and no assertion fired. The FMA miter ran 210 M vectors × 2 latencies with 0 mismatches.
- **VFPU and matrix oracles.** The `rtl_oracle.py` (400 per op) and `matrix_oracle.py` (ABUF 0 and 1) outputs are byte-identical to r41s2-int's. The checksums are in `oracle_md5.txt`.
- **Cycles.** Both rounds are identical to round 4 on every core suite, `r41-core-boundary`, the multicore suites and the 16 pipeline kernels. Against r41s2-int the total is therefore still +0.4–0.52 % on the core benches and +0.34 / +0.65 % on the pipeline kernels.
- **Killed simulator.** In round 6 another agent killed one `r40-pipeline-checks` simulator, which showed up as `DEVICE_LOST` in `gemm_odd_steps`. I reran the target: it passed, with cycles identical to round 5.

### Screens

Cells give the worst slack (ns) / endpoints below +1.4 ns.

| Tile 1 | r4 `2549feb` | r5 `9f22689` | r6 `3643bfb` |
|---|---|---|---|
| Transfer engine (internal) | +1.193 / 33 | +1.322 / 2 | ≥ +1.5 |
| W forward | +1.320 / 1 | ≥ +1.5 | ≥ +1.5 |
| F04 writeback | ≥ +1.5 | +1.384 / 1 | +0.679 / 61 † |
| Late VALU stage | +1.124 / 9 | ≥ +1.5 | +1.289 / 22 |
| VFPU | +1.161 / 25 | +0.896 / 145 (FMA stage 0) | +1.076 / 15 |
| MTX | +1.054 / 20 | +1.259 / 7 | +1.136 / 16 |
| VRF bank read | +1.500 / 0 | ≥ +1.5 | +0.641 / 72 † |
| **Tile worst** | +1.054 | **+0.896** | +0.398 † |

| Tile 0 | r4 | r5 | r6 |
|---|---|---|---|
| Transfer engine, W forward | ≥ +1.5 | ≥ +1.5 | ≥ +1.5 |
| F04 writeback | +1.431 / 0 | +1.099 / 3 | +1.236 / 7 |
| Late VALU stage | +0.885 / 99 | ≥ +1.5 | ≥ +1.5 |
| VFPU | +0.690 / 260 | +1.006 / 323 (FMA stage 0) | +1.259 / 9 |
| MTX | +0.810 / 222 | +0.973 / 38 (bank → `a_q`) | +1.388 / 1 |
| NUCA slice (F07) | +1.042 / 39 | +1.453 / 0 | +0.798 / 366 † |
| **Tile worst** | +0.690 | **+0.973** | +0.798 † |

**Kernel synthesis.** WNS +3.914 ns in both rounds (`cvt_x` → `dly_v[1]`). LUTs: 812,043 in r5, 815,245 in r6.

† These paths are 87–92 % routing, and their logic did not change in round 6:
- `x_sa_fw` → `adv_q`, 12 levels, 8.4 ns of routing;
- `x_lm_fw` → VALU mask → `w_vdata`, 6 levels, 8.6 ns of routing;
- `pf_tag` → bank `s1_vld`;
- slice LUTRAM address.

Round 6's 1,536 extra registers moved placement. Between rounds that change only a few hundred registers, the same unchanged families move by ±0.5 ns. One routed screen per round cannot show a family at ≥ +1.4 ns with any confidence.

**Assessment.**
- **Logic.** Every family targeted in rounds 1–6 is now logic-shallow. The paths below +1.4 in the latest two runs are either:
  - route-dominated, and they move between runs;
  - or in the FMA stage-0 input, which round 6 fixed and which is otherwise generated XLS logic.
- **Best measured tree.** Round 5 (`9f22689`) is the best balanced: tile worst +0.896 / +0.973 ns, with no family below +0.89. Round 6 removes the FMA operand-routing family but measured worse on unrelated routing paths.
- **Recommendation.** Use `9f22689` for the next image, unless the full-run calibration shows FMA stage 0 as an image limiter. In that case use `3643bfb`.
