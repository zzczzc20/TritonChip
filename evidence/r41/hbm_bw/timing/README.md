# R41 K8: matrix VRF response timing register

> Binding: [`doc/engineering_discipline.md`](../../../../doc/engineering_discipline.md). All timing numbers below are Vivado measurements; all cycles are RTL simulation. No board, tiled2, or full-link run was performed.

## Scope and identity

The generic, default-off `MTX_VRF_REG` parameter registers banked VRF port 3's matrix response data and valid. It does not change the ISA, compiler, instruction issue/scoreboard, matrix state machine, or numerics. Reset and `launch_valid` clear the valid bit. While the matrix unit is requesting data, the request sent to the VRF is suppressed when either the original or registered response is already valid, preventing a duplicate read.

- Implementation commit: `b87a0cc3090762d49b412c5adfb887d592fc3306` (`Add optional matrix VRF response register`), parent `552262c013f955efc6522e16e6922227d3687dfd`.
- Default: `MTX_VRF_REG=0`. The candidate image differs from the G2C4 baseline only by `MTX_VRF_REG=1`.
- Core implementation: `rtl/core/tnca_core.v:85-87` and `rtl/core/tnca_core.v:920-933`; the request suppression is at `rtl/core/tnca_core.v:1073`.
- Matrix consumer: `rtl/core/tnca_matrix.v:226-228`; MAC operand capture is the `g_in_reg.a_q_reg` bank.
- Baseline image SHA256: `8c3f7219c377354f4b48dc09a257971c852297720c5866a17e58e3caafe166d9` (`../g4/images/image_g2c4.tcl`). Candidate image SHA256: `296a6d4136b99cd1dc1ed2601186b7c978741b781713b9f9ffc3f15a4d70df87` (`image_g2c4_mtx_vrf_reg.tcl`). Both use four contexts and `core-vrf-sync-v5-x8-g2c4`.

## Previously measured critical families

These are the pre-change routed measurements used to select the fix, not predictions from this change. OOC tile screens contain no SLR crossing; the only crossing listed here is from the run-1 full image.

| Source | Family and exact worst registers | Slack at 10 ns | Levels and relevant data-net fanout | SLR crossing |
|---|---|---:|---|---|
| run-1 full image, tile0/SLR0 | F04 writeback `of_d1` → `w_vdata` | +0.042 ns | 23 | no |
| run-1 full image, tile0/SLR0 | F07 NUCA `u_slice/cur_ms_reg_rep` → `data_mem` write ports | +0.065 ns | 15–16 | no |
| run-1 full image, tile0/SLR0 | F09 VRF bank `mem_reg` → `u_mac a_q` | +0.079 ns | 16 | no |
| run-1 full image | quiescence tile0 `u_mtx/st_reg` → MMIO control registers | +0.180 ns | 7–10 | **SLR0 → SLR1** |
| run-1 full image, tile1/SLR1 | VFPU `of_d1` → `u_vfpu g_fma s0_q` | +0.184 ns | 18 | no |
| run-1 full image, SLR1 | DCO local TMU `tl_class_id_q` → `lv_valid` | +0.281 ns | 12 | no |
| G2 combined tile0/SLR0 OOC | `g_vrf_banked.u_vrf/g_bank[0].u_bank/mem_reg_7/CLKARDCLK` → `u_mtx/g_mac[7].u_mac/g_in_reg.a_q_reg[31]/D` | +0.603 ns | 15; route 7.128 ns | no |
| G4 s4 tile0/SLR0 OOC | `u_core/x_func_reg[5]/C` → `u_core/u_vfpu/acc_reg[7][28]/D` | +0.423 ns | 4; `func[5]` fanout 795, next net 128; route 9.197 ns | no |
| G4 s4 tile0/SLR0 OOC | `u_core/w_sdata_reg[0]/C` → matrix row-op `g_mac[3]...a_q_reg[19]/D` | +0.560 ns | 13 | no |

The VFPU path is `x_func` in `rtl/core/tnca_core.v` feeding `func` and `acc` in `rtl/core/tnca_vector_fpu.v:54,130`; it is not modified by this task. Sources: `evidence/r41/stage2/run1/README.md:63-77`, the G2 routed path report under `../resources/tile_screens/tile0_g2all_8057dea1fb61/`, and `../g4/screens/s4_64e3c71d7f39/tile0_timing_routed_head.rpt:204-261`.

## Directed RTL lockstep

On GB10 `gx10-6646`, `make core-mtx-vrf-reg` passed all eight original/candidate pairs (normal and out-of-order memory, `MTX_ABUF=0/1`, two matrix programs). The test compares instruction, accumulator-write, `src_free`/done events and self-checked numerical results. It also checks issued/hazard/writeback/squash counts and accounts for the cycle delta as accepted matrix VRF reads plus the measured LSU-stall delta.

| ABUF | Memory | Program | Original cycles | Candidate cycles | Delta | Accepted reads | LSU-stall delta |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | normal | t11 | 4731 | 6091 | +1360 | 1360 | 0 |
| 0 | normal | t25 | 1722 | 2266 | +544 | 544 | 0 |
| 0 | out-of-order | t11 | 5121 | 6477 | +1356 | 1360 | −4 |
| 0 | out-of-order | t25 | 1980 | 2522 | +542 | 544 | −2 |
| 1 | normal | t11 | 3515 | 3627 | +112 | 112 | 0 |
| 1 | normal | t25 | 1242 | 1290 | +48 | 48 | 0 |
| 1 | out-of-order | t11 | 3913 | 4008 | +95 | 112 | −17 |
| 1 | out-of-order | t25 | 1509 | 1552 | +43 | 48 | −5 |

The out-of-order LSU difference comes from external-memory timing reordering and is reported rather than hidden. Raw logs, including the two earlier failed attempts and the final passing run, are in `rtl-directed-logs.tgz` (SHA256 `4459189fcb1a687d56039061da34376ae4acbebe6b67e401b121db49f8a0e16f`). The first attempt did not account for the LSU delta; the second passed the comparisons but failed provenance finalization because its exported tree had no `.git` metadata.

## G2C4 x86 synthesis and tile0/SLR0 OOC screen

Host: x86 `ee4e138`, Vivado 2023.2, device `xcu280-fsvh2892-2L-e`, clock 10 ns. Both jobs used the same `b87a0cc3` RTL and at most four threads. Kernel synthesis commands, run after `source ~/x86_link_env.sh`:

```sh
cd ~/r41-k8-tim/screens/baseline
vivado -mode batch -nojournal -log synth.log \
  -source ../../src/flow/fpga/synth_r40_multicore.tcl \
  -tclargs ../../src ../../src/image_g2c4.tcl out
cd ~/r41-k8-tim/screens/mtx-vrf-reg
vivado -mode batch -nojournal -log synth.log \
  -source ../../src/flow/fpga/synth_r40_multicore.tcl \
  -tclargs ../../src ../../src/image_g2c4_mtx_vrf_reg.tcl out
```

Each resulting checkpoint was screened independently:

```sh
vivado -mode batch -nojournal -log screen.log \
  -source ../../src/flow/fpga/screen_r41_tile.tcl \
  -tclargs out/synth.dcp 0 0 tile0
```

### Resources

Kernel synthesis top (`tnca_multicore_hbm_xo`):

| Configuration | LUT | FF | Logic LUT | LUTRAM | SRL | RAMB36 | RAMB18 | DSP | Synthesis WNS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 768110 | 217312 | 718135 | 49756 | 219 | 192 | 45 | 520 | +4.031 ns |
| candidate | 766426 | 218848 | 716463 | 49744 | 219 | 192 | 45 | 520 | +4.031 ns |
| delta | −1684 | +1536 | −1672 | −12 | 0 | 0 | 0 | 0 | 0 |

Routed tile0/SLR0 OOC:

| Configuration | CLB LUTs | CLB Registers / FF | CLB | Block RAM tiles | DSP | Setup WNS / TNS | Setup failing | Hold WHS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 239391 | 64379 | 40858 | 39.5 | 171 | +0.794 / 0.000 ns | 0 | +0.004 ns |
| candidate | 237768 | 64758 | 39889 | 39.5 | 171 | +0.835 / 0.000 ns | 0 | +0.017 ns |
| delta | −1623 | +379 | −969 | 0 | 0 | +0.041 / 0.000 ns | 0 | +0.013 ns |

The LUT and CLB reductions are net synthesis/placement results, not claimed as a deterministic saving attributable to the added register. The candidate's worst path moved away from the VRF→matrix family to `u_slice/est_reg[3]/C` → `u_slice/data_mem_reg_r2_64_127_49_55/RAMB/I`: 19 levels, 9.089 ns data delay, `est[3]` fanout 459 and a LUTRAM address net fanout 2048. The baseline worst was `g_bank[1].u_bank/mem_reg_7/CLKARDCLK` → `g_mac[2].u_mac/g_in_reg.a_q_reg[16]/D`: 14 levels, 9.140 ns, with data-net fanouts 16 and 17 in the MAC selection cone.

### Direct check of the cut path

A subsequent read of the candidate's routed checkpoint found 512 `g_mtx_vrf_reg.data_q_reg` cells and one `valid_q_reg`. Explicit `report_timing -from/-to` queries measured:

| Segment | Worst source → destination | Slack | Levels | Data delay |
|---|---|---:|---:|---:|
| VRF → response register | `g_bank[0].u_bank/mem_reg_1/CLKARDCLK` → `g_mtx_vrf_reg.data_q_reg[93]/D` | +6.193 ns | 5 | 3.741 ns |
| response register → MAC | `g_mtx_vrf_reg.data_q_reg[507]/C` → `g_mac[9].u_mac/g_in_reg.a_q_reg[15]/D` | +3.030 ns | 9 | 6.952 ns |

Thus the original 14-level, +0.794 ns VRF→MAC path is split into two registered segments in this routed candidate. The first query attempt failed because its Tcl file had not yet arrived on the x86 host; its log and the successful rerun are both retained.

Raw kernel/tile utilization and timing reports, top 2000 paths, failing-endpoint files, synthesis/P&R logs, and both query attempts are in `x86-g2c4-reports.tgz` (SHA256 `5ef8675bfb49a14e92c12bbdc81c17c92e3680575976ed6bd57fe210fe8c2b71`). DCPs are not committed.

## RTL/compiler gate and default-code audit

On GB10 `gx10-6646`, the sequential R40/R41 gate ran with `R1_STRICT=1` and the pinned Torch/LLVM environment at `b87a0cc3`. All 28 non-compiler Make targets and the separate multicore MMIO test returned 0, including core profiles, slow/out-of-order memory, G2C4, boundary, top/xpose, LSU, FMA, lint/unit, multicore/grid/cache/coherent/barrier/parameter, compiler-RTL, Triton end-to-end, pipeline/line-AXI and DMA tests.

The first strict `make compiler` failed because `doc/support_matrix.md` was stale. After fast-forwarding the unrelated upstream regeneration (`7295f6a7`), the next attempt failed because the source rsync had removed the previously built `tnca-opt`; `make tnca-opt -j3` followed by strict `make compiler` passed on merged source `0e738be9`. Both failures and the passing log are preserved.

`default_change_audit.py run --jobs 2` compiled all 113 points independently on the `552262c0` control and `b87a0cc3` candidate, with launches skipped and device sessions forbidden. Both runs passed 113/113. `default_change_audit.py compare` produced **113 unchanged, 0 changed, 0 not compared**. The first audit invocation failed before running points because its inline `software_tree` command produced an empty-tree hash; the failed logs and corrected rerun logs are retained. This is a compile-only code/launch-structure audit, not a numerical or performance result.

Raw gate and audit reports are in `gb10-regression-audit.tgz` (SHA256 `9574b17f7ad57ac966397cc07e5f578d336da5eb739ed2622dacb698e063680b`), including the failed compiler/audit attempts and the initial 2 MiB RTL capacity failures.

## Run-2 integration and gate plan

The current integration head contains terminal 9's `255d2ad0ac15bd5a51fb869e04e6a130c3b0e205`, which restores at most eight beats per unit for hybrid narrow-LSU accesses with wide tags while leaving the default configuration unchanged. Its two-point numerical RTL pass and merged add/clone measurements of 135227/90167 cycles, plus its hybrid/line/NoC/home and strict compiler regressions, are terminal 9 handoff results, not independently reproduced by K8. Terminal 9 did not run the 113-point audit or claim the complete gate.

The four in-flight K8 113-point experiments are unchanged. Before the run-2 image is accepted, an independent integration tree must run lint/unit, hybrid/line/NoC/home, the core R40/R41 suites, strict compiler, and directed default-configuration and wide-tag eight-beat tests. That exact head also needs the 113-point default-code audit and complete 113-point numerical/cycle evidence. The audit at `b87a0cc3` and the in-flight K8 runs do not cover `255d2ad0`; a HostSession audit is compile-only code/launch-structure evidence, not device acceptance. The integrated head additionally needs a new tile timing screen. No unapproved full-link or card run is planned.

## Remaining regression evidence

The full original/candidate 113-point RTL cycle comparison is **still running** and is not reported as complete here. An initial 2 MiB attempt returned `capacity-exceeded`; a 256 MiB GPT-2 probe exceeded its two-hour local task timeout while the remote RTL simulation continued. These attempts must not be read as measured cycles. Final 113-point results and their raw logs will be added when the runs finish.

No claim is made about full-image timing closure. These OOC screens have no shell, other tiles, static-region density, or SLR crossing.
