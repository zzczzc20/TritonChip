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
