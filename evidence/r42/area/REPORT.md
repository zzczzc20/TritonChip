# R42 area worker: final report (written by the lead from the worker's message; the harness refused the worker's own REPORT.md)

Branch `R42-area` (merged into `R42`), commits `67ffb809`, `706538c0`, `e3680b7c`, `521ea7e8`. Domains: **OOC** = out-of-context place and route
(xcu280, Vivado 2023.2, x86 host, ideal clock, no pblock); **synthesis** = kernel synthesis; **routed** = the run-3 routed checkpoint, read only;
**derived** = arithmetic on recorded figures. Nothing here is a board number.

Not finished (the x86 slots were held for hours by another worker's kernel synthesis and tile screens; the worker stopped its queue): hybrid-LSU OOC at
10 / 6.67 / 5 ns, the tile-0 screen at 6.67 ns, the cycle cost of the latency-adding fixes. Rerun: `queue_t2.txt` with `tools/r42/area/run_queue.sh`.

## Result

1. fp32 FMA, the shared MAC lane's back end and the vector FPU's int/float converter were rebuilt and are bit-identical to the generated RTL. Savings are small:
   FMA 1,497 -> 1,409 LUT routed (-5.9 %); MAC lane 3,146 -> 2,966 (-5.7 %); converter lane 504 -> 418 (-17 %); tile 0 in synthesis 253,898 -> 248,890 LUT
   (-2.0 %, flatten none) or 253,205 -> 249,588 (-1.4 %, flatten rebuilt); with the converter about -2.5 % / -2.0 %.
2. DSP48E2 mapping of the adders does not pay: three 49-bit sum adders on DSPs (latency 3) give an FMA of 1,396 LUT / 5 DSP against 1,351 LUT / 2 DSP for
   the existing XLS latency-3 FMA; in the MAC lane the same three DSPs change the LUT count by -2.
3. Cores per SLR stay at 1: a core of about 138k LUT would be needed for 5 cores (derived). Arithmetic mapping is worth 2-2.5 % of a core.
4. No single unit bounds the clock. In the run-3 routed image 18,548 kernel-clock endpoints have a routed path longer than 9 ns and 89,603 longer than 8 ns, in every unit.

## Built

New stages in `rtl/core/`: `tnca_fmax_mul`, `tnca_fmax_alignp`, `tnca_fmax_align`, `tnca_fmax_add`, `tnca_fmax_sel`, `tnca_fmax_norm`, `tnca_lzc81`, `tnca_fmax_round`,
`tnca_fmax_fma` (five stages, selectable pipeline cuts), `tnca_cvt_int` (int/float lane). Parameter plumbing in `tnca_fp32_fma`, `tnca_mac_lane`, `tnca_matrix`,
`tnca_vector_fpu`: new parameter `BE` selects the area back end; `MAC_CUTS` widened to 6 bits; `FMA_IMPL = 1 + 16*BE (+256 for the converter)` reaches every lane through the
existing image parameter; default 0 keeps the generated RTL. Tooling in `tools/r42/area/` (OOC flow, kernel-synthesis wrappers, census scripts).

## Verification (RTL simulation)

| Check | Size | Result |
|---|---|---|
| Stage miters against the generated stages (align, norm, round, whole FMA, struct level) | 9 classes x 1M FMA vectors + 4 classes x 1M arbitrary round structs, SHIFT 0 and 1 | 0 mismatches |
| Cycle-level FMA miters at latency 2 and 3 | 9 classes x 1M, six configurations | 0 mismatches |
| MAC lane miters, all modes, struct and result level | 3 classes x 3 modes x 1M, BE 5 and 13, three cut sets | 0 mismatches |
| Exact Fraction oracle, `make fma-deep-area`, BE 13 and 14, final tree | 516,240 vectors | 0 failures |
| `tnca_cvt_int` against `cvt_int_f` | every 32-bit word for fn 0/1 (rm 1) and fn 2/3 (rm 0..3), ten runs of 4.29e9, plus 2e9 random triples over all 256 function codes and 8 rounding modes | 0 mismatches |
| Mutation sensitivity of the miters | 27 single-point changes | 24 caught; the 3 survivors are equivalent by construction |
| `make core-run2`, default `FMA_IMPL=1` | 90 program runs, in-order and reordered memory | 90 PASS |
| `make core-run2`, `FMA_IMPL=465` (BE 13 + converter) | same 90 runs | 90 PASS; every cycle count equals the default's |

The default build reproduces the R42-img worker's tile 0 exactly (253,898 LUT). Two small RTL edits made after that synthesis (converter select in `tnca_vector_fpu`, `% 16` in
`tnca_fp32_fma`) add only a constant-false select or unused wires; the 90-run core-run2 default pass covers that tree.

## (A) Area

Tile 0 by unit (kernel synthesis, hierarchy kept, run-3 image parameters): matrix engine 61,552 (24 %), vector FPU 49,907 (20 %), NUCA home slice 39,488 (16 %, incl. 9,902 LUTRAM),
vector integer ALU 23,173 (9 %), banked VRF 21,270 (8 %), transfer engine 19,503 (8 %), hybrid LSU 19,338 (8 %); tile total 253,898 (tiles 1, 2: 251,485 and 252,127).
Matrix engine: 16 MAC lanes x 3,203 (51,248), accumulator file 6,293, other 4.0k. Vector FPU: 16 FMAs x 1,545 (24,720), converters 7,520, SFU 2,330, own logic 15,337.

Unit results (OOC, routed; LUT / DSP / WNS at 10, 6.67, 5 ns):

| Unit (latency) | LUT | DSP | WNS 10 | WNS 6.67 | WNS 5 |
|---|---:|---:|---:|---:|---:|
| FMA generated IMPL0 (2) | 2,269 | 2 | +3.600 | | |
| FMA IMPL1, today (2) | 1,497 | 2 | +4.151 | +0.941 | +0.309 |
| FMA IMPL1, XLS latency 3 | 1,351 | 2 | +4.720 | +1.219 | +0.441 |
| FMA new, generated align (2) | 1,409 | 2 | +4.371 | | +0.138 |
| FMA new, area align (2) | 1,411 | 2 | +3.864 | | +0.329 |
| FMA new, adders on DSP (3) | 1,396 | 5 | +4.115 | +1.315 | +0.439 |
| MAC lane today (5 cuts) | 3,146 | 8 | +2.527 | +0.333 | -0.664 |
| MAC lane BE 13 | 2,966 | 8 | +2.156 | +0.466 | |
| MAC lane BE 13 + register behind adders | 2,964 | 11 | +2.260 | +0.354 | -0.647 |
| int/float lane `cvt_int_f` | 504 | 0 | +5.08 | | |
| int/float lane `tnca_cvt_int` | 418 | 0 | +5.44 | | |
| SFU, one four-function slice | 2,300 | 14 | +4.335 | | |

Tile totals (kernel synthesis; tile 0 / 1 / 2 LUT): default 253,898 / 251,485 / 252,127 (flatten none), 253,205 / 251,554 / 252,116 (flatten rebuilt); BE 13 248,890 / 246,477 / 247,119
and 249,588 / 247,940 / 248,499; change per tile -5,008 (-2.0 %) and -3,617 (-1.4 %); whole kernel 814,086 -> 799,062 and 814,467 -> 803,619. Unit-level numbers depend on the flow (stage-3
shifter 463 LUT hierarchy kept, 351 flattened); use the tile totals.

DSP inference rules found: an adder reaches a DSP only when a register follows it; a static registered 48-bit add gives 1 DSP and 0 LUT; a combinational adder stays in carry
chains; a dynamic add/subtract with carry-in stays in carry chains; `a - b - cin` uses 2 DSPs but keeps the carry-in in logic; three static registered adders give 5 DSPs, 96 LUT.

## (B) Clock

Routed run-3 image (kernel clock, setup WNS +0.111 ns). Endpoints below +2.44 ns slack, three tiles summed (slack s at 10 ns is a path of about 10 - s ns):

| Unit | < 0.5 ns | < 1 ns | < 2 ns | Worst path |
|---|---:|---:|---:|---|
| NUCA slice | 276 | 6,599 | 34,699 | reset_sync -> `data_mem` LUTRAM WE, 1 level, 9.28 ns, 98 % net |
| device/control (cross-SLR) | 47 | 2,970 | 20,928 | LSU `fast/valid_reg` -> `control/rsp_status`, 14 levels |
| hybrid LSU | 372 | 6,580 | 8,503 | `fast/rr_reg` -> `out_vector`, 25 levels, 9.85 ns, net 76 % |
| core control/pipeline regs | 32 | 768 | 6,416 | `w_vdata`, 10 levels; pc/insn regs 16-23 levels |
| banked VRF | 46 | 328 | 5,836 | `of_acc_q` -> bank `s1_byp_mask`/`s1_vld`, 16-18 levels |
| transfer engine | 0 | 55 | 4,668 | `t_ps` -> `cm_cnt`, 18 levels |
| matrix engine | 57 | 679 | 4,423 | `w_sdata`/`mm_reg` -> MAC lane multiplier, 15 levels |
| vector FPU | 70 | 468 | 2,925 | reset_sync -> `acc_reg` CE/R, 1 level, 98 % net; FMA `s0_q`; SFU `cnt` -> exp2 table |
| barrier | 0 | 53 | 666 | `u_srf` valid -> `done_epoch` CE, 18 levels |
| vector integer ALU | 0 | 2 | 161 | VRF `s1_dst` -> `q_pu` DSP |
| **All kernel clock** | **900** | **18,548** | **89,603** | |

At least 144,674 kernel endpoints are below +2.44 ns (census capped; counts at 150 and 200 MHz not taken). Per-tile worst paths: `clock/top_paths_by_unit.txt`. The FMA's first stage is 4.67 ns
in OOC but at least 7.6 ns in the image (derate 1.6-2.0). In OOC the FMA meets 5 ns; the MAC lane meets 6.67 ns but not 5 ns (stage 1 5.65 ns, stage 2 5.4 ns).

What each family needs (cycle costs derived, not measured): reset fan-out into `acc_reg`, `data_mem` WE and accumulator valid -> per-tile registered reset tree, 0 in operation; hybrid-LSU `fast`
select -> split lane merge and output mux over two stages, +1 cycle on every load response; matrix operand capture into MAC lane stage 1 -> one more operand register, +1 cycle per matrix command
(0.6 % of a 168-cycle K16 command); VRF bypass/forward selects -> re-time the forward mask, +1 cycle on vector RAW forwarding; slice LUTRAM writes -> registered address or BRAM/URAM, +1 cycle on a
slice hit; core front end, barrier, transfer counters -> deeper issue path, +1 cycle issue/branch penalty; FMA first stage -> latency 3, +1 cycle per dependent vector FP op (free in the matrix path); MAC
lane stages 1 and 2 at 5 ns -> `MAC_CUTS` bit 1, +1 cycle per matrix command; SFU table -> one more stage, SFU latency 6 -> 7; LSU valid -> MMIO status -> register it, 0.

Reading: 125 MHz needs every path within 2 ns of the limit fixed (89,603 endpoints); 150 MHz needs more than 144,674. After the first three fixes (LSU select, reset tree, matrix operand capture) the next
endpoints are the slice, VRF and barrier at +0.2 to +0.5 ns and the control registers at +0.2 to +0.6 ns.

## (C) Cores per SLR (derived)

Inputs: routed run-2 link SLR use (CLB 95.56 / 90.66 / 88.06 %); highest packing that routed in an image 96.74 % (run 1); routed tile LUTs = 0.983 x synthesis tiles.

| | SLR0 (HBM, outlets) | SLR1 | SLR2 |
|---|---:|---:|---:|
| LUT at 96.74 % CLB | 325.9k | 361.0k | 319.1k |
| Non-core LUT today | 72.3k | 85.2k | 37.6k |
| Room for cores | 253.6k | 275.8k | 281.5k |
| Cores of today's 249.6k | 1.02 | 1.10 | 1.13 |
| Core size for 2 in the SLR | <= 126.8k | <= 137.9k | <= 140.8k |

1 core per SLR fits. 4 cores (2 in SLR2) need a core of 140.8k LUT, 5 cores 137.9k, 6 cores 126.8k: reductions of 44, 45 and 49 % from 249.6k. Halving the lane counts removes about 60k (linear scaling of
per-lane figures) and leaves about 190k, still above 141k: a 138k core is a different core. DSP supply is not the limit (818 of 9,024 used).
Target LUT per core about 247-248k with this work; achievable clock about 100-110 MHz for this netlist, roughly 120 MHz after the top nine families are fixed (derived); cores per SLR: 1.

## Failed or not done

- DSP adders: no LUT or timing gain (FMA +45 LUT against the XLS latency-3 instance at equal WNS; MAC lane -2 LUT, +3 DSP). Kept as `CUTS[2]` / `MAC_CUTS[5]`, off by default.
- Not built: multiplier-based shifters on DSPs; NUCA arrays and accumulators on URAM (a synchronous read adds a cycle to the slice's hit path); the vector ALU's 120 pairwise comparators for the integer
  min/max reduction (about 3.8k LUT per core, derived only); the rest of the vector FPU's own logic apart from the converter.
- Setbacks (own, fixed or discarded): a kernel synthesis ran alongside the 16 GB routed-checkpoint load and hit memory pressure (synthesis killed and rerun); two of the worker's own commands killed their own shell
  because a `pgrep` pattern matched their command line; the output directory of a running census job was deleted (job died at its summary step, rerun).

## Remaining ideas, ranked by expected gain

1. Re-pipeline the families in section B: the only route above 110 MHz, about a cycle per fix.
2. Cut the vector ALU's reduction comparators, about 3.8k LUT per core.
3. Move slice and accumulator arrays to URAM, about 12k LUT per core of LUTRAM plus read muxes.
4. Rewrite the rest of the vector FPU's own logic (15.3k LUT; the converter alone saved 1.4k per tile).
