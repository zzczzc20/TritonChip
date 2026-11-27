# R42 dse: TLM design-space exploration, final report (written by the lead from the worker's message; the harness refused the worker's own REPORT.md)

Branch `R42-dse` (merged into `R42`), last commit 6924848e. **Domain of every number: SystemC/TLM model screens applied as ratios to board cycles; none is a board number.**
Tables: `tables.txt`, `tables_screen.txt`, `tables_full.txt`; data `col/` (raw model runs), `cfg/` (56 configurations), `full.json`; costs `lever_costs.txt`; failed first run `failed_first_validation/`;
commands in order: `commands.txt`.

## 4x verdict
On the 145-point wall-time geometric mean, 4x is reachable in the model only with about a 200 MHz clock, or 6 cores at 200 MHz; no combination that was run fits the recorded physical evidence. It is not reached at
150 MHz or below with 3 to 4 cores (best 3.3).

## What changed
- Model keys in `tools/performance_harness/systemc/`, all default-off: `lsu_hybrid`, `lsu_fast_slots`, `lsu_slow_units`, `issue_vector_cycles`, `issue_two_src_cycles`, per-class `issue_{vint,vfp,cvt,shuffle,sfu,vlsu,vred}_cycles`,
  `core_mem_ports`, `issue_free_mode` 4 and 5, `run_graphs.py --cache-seed`.
- Default-off identity: the new binary matches a binary rebuilt from `origin/R42` sources on 27 of 27 tiny points; the first run of the unpatched binary differs on `op.transpose/tiny` (1324 vs 1298), a TNCA compile-cache effect both binaries show.
- Scripts `tools/r42/dse_*.py`, `dse_gb10/*.sh`.

## Step 1: validation (tiny/small/medium direct; large, hbm, xl by direct run, proxy or wave model)
- First validation, kept: 77 tiny/small/medium points on the `cal` model: geometric mean model/board 0.807, median |error| 0.206, 38 of 77 within 20 %.
- Diagnosis: on 24 tiny points against single-core RTL with zero-latency memory TLM/RTL was 0.84 (non-GEMM 0.76): the shortfall is in the core pipeline, not memory.
- Fix: a hybrid LSU model (fast line path plus 2 word units, mode drains); every vector instruction except shuffles then takes 2 issue cycles, calibrated on RTL numbers only. Single-core RTL median error 0.085.
  Board holdout (77 points): geometric mean 0.983, median |error| 0.132, 52 of 77 within 20 %.
- All 145 points: 141 modelled (4 xl composites have no stand-in). Geometric mean 0.956, median |error| 0.158, 87 of 141 within 20 %. Per tier: tiny 0.995, small 0.988, medium 0.967, large 0.903, hbm 0.786, xl 1.010.

| Verdict | Families (geometric mean model/board) |
|---|---|
| Usable | add fp16 0.84, argmax 1.03, conv2d 0.98, gelu_tanh 0.99, gemv 0.88, gpt2_block 1.06, layernorm 0.92, linear 1.03, llama_decode 0.86, llama_prefill 0.96, matmul fp16, mlp 1.03, resnet_block 1.00, rmsnorm 0.89, rmsnorm_residual 1.07, softmax 0.99, fp8_act_quant 0.81, fp8_layer_decode 0.85 |
| Not usable, model too fast | rope 0.56, transpose 0.51, copy 0.58, silu 0.73, swiglu 0.75, attention_decode 0.81 (borderline) |
| Not usable, model too slow | attention_prefill 1.35, bmm 1.20, cumsum 1.49, embedding 1.30, fp32 matmul 1.42 to 1.47 |
| fp8_linear | small shapes (direct, M up to 16) 1.02 to 1.05 (board-validated); large-M wave-model points 1.24 to 1.35, family not usable overall |

Missing features, not added: the 2-cycle vector issue over-penalises FMA, matrix-epilogue and fp32 kernels; the too-fast families probably miss a shared SFU/FPU completion-port stall (silu, swiglu) and the word-path timing of
gathers (transpose, embedding); none probed in RTL. The mechanism of the 2-cycle issue (VRF read-ahead overlaps only when R is idle) is an unverified reading of `tnca_core.v`; the calibration is empirical.

## Step 2: set-up of the levers
- Rule: `pred_cycles = board_r2 x m(head software, cfg) / m(round-2 software, base)`; wall time = `pred_cycles / f`. Round-2 board 1.556 (reproduced exactly).
- Model-predicted round 3 (head software, today's hardware): S145 central **2.07**, low 1.85; by tier tiny 1.55, small 2.23, medium 2.45, large 2.30, hbm 2.84, xl 1.80. (Of the model's 1.33x software factor, embedding alone contributes 1.13x;
  embedding is a not-usable family: the model puts its software gain at about 20x, the copy worker's TLM predicts 18.8x.)
- Cores: one NUCA slice of 16 KiB, one HBM outlet and one mesh node per core; `GEMM_MIN_PROGRAMS` (12) and `TILED_MIN_PROGRAMS` (8) left at their 3-core values.
- Clock: memory round trip fixed in ns (23 cycles at 100 MHz, a fit of the board calibration); outlet return 5 RTL cycles + 8 fitted cycles fixed in ns; `clkN` keeps RTL pipeline depths in cycles, `clkNp` scales them by 1+(r-1)/2, r = f/100 MHz; at 250 MHz the outlet issues one line per 2 cycles.
- Sensitivity: the hardware effect is raised to a power e (low/central/high 0.5/1.0/1.2 on a gain, opposite on a loss), bracket from the three lever-versus-RTL checks (G lever RTL/model 1.10-1.18, N-blocked tile 0.33-0.86, LINE_HALF 0.84); the low case also applies
  hardware effects only to usable families and halves the software log-gain on the others.
- LINE_HALF check: model le2/le0 0.742 against RTL three-core new/old 0.778 on 9 points: log-gain ratio RTL/model 0.84.
- Coverage: tiny/small/medium run for 26 levers; heavy-tier direct runs for 7 (c6, c6o3, clk200, vi1, memideal, sets128, sets512); full 145-point bundles for c6, clk200, vi1, k1-k9; the rest inherit by family. Defined but not run: lane2, port2, noc1, lsuf16, lsuf4, sfu4, sfu16, vrf8, te16, te64, le2x, g4, mpf, s1, sets64, clk250p.

## One-at-a-time effects on the 145-point wall-time geometric mean (model, central)

| Lever | Effect |
|---|---|
| Clock 200 MHz | 1.79 (1.60 with pipelined latencies); 150 MHz 1.46; 250 MHz 2.27 |
| Cores 6 | 1.68 (heavy tiers 1.9, small/medium 1.8, tiny 1.03) |
| Cores 4 | 1.17 (tiny/small/medium run, rest inherited) |
| Single-cycle vector issue (`vi1`) | 1.13 (1.10 to 1.16 across tiers) |
| LINE_HALF (le2 vs the line engine le0) | 1.07 (about 1.06 after the RTL-measured 0.84) |
| 2-wide issue (upper bound) | 1.06 |
| Hardware loop / zero-cost branch | 1.03 each |
| NUCA slice 4x | 1.05 (heavy 1.04); 16x heavy-only 1.07 |
| Tensor-engine slots 8 to 32 | 1.03 |
| Half-latency HBM | 1.03 |
| Next-line prefetch | 1.016 |
| SFU 2 slices | 1.014 |
| 4 word units | 1.013 |
| Ceiling of all memory-side levers (`memideal`, unphysical) | 1.23 (heavy 1.29) |

At or below 0.5 %: K-fusion G=8 (+0.5 %), macro-command issue (-0.2 %), VRF bank conflicts removed (+0.2 %), 6 outlets at 3 cores (+0.1 %), 64 outstanding lines/MSHR (0.0 %). 6 cores with 3 outlets match 6 cores with 6 outlets to 0.1 % on the 21 directly run heavy points.

## Combinations (all 145 points, 3 cores unless stated)

| Config | S145 low / central / high | Central with round-2 software |
|---|---|---|
| k4: 150 MHz pipelined + vi1 | 2.54 / 2.83 / 3.03 | 2.13 |
| k7: k4 + 2-wide issue + zero-cost branch | 2.72 / 3.29 / 3.50 | 2.47 |
| k3: 4 cores + 150 MHz pipelined + vi1 | 2.60 / 3.20 / 3.46 | 2.41 |
| k6: 200 MHz pipelined + vi1 + 2-wide + zero-cost branch | 3.38 / 3.85 / 4.21 | 2.89 |
| k5: 6 cores + 200 MHz pipelined + vi1 | 3.88 / 5.64 / 6.17 | 4.24 |
| k1: 6 cores + 200 MHz | 4.14 / 6.45 / 7.09 | 4.85 |
| k2: k1 + vi1 | 4.25 / 7.19 / 8.05 | 5.40 |
| k8: 6 cores + 200 MHz pipelined + vi1 + 2-wide + branch + NUCA 4x | 4.20 / 7.25 / 8.21 | 5.45 |
| k9: 8 cores + 250 MHz pipelined + the rest | 5.49 / 10.84 / 12.83 | 8.15 |
| k10: k9 with zero-latency memory (unphysical ceiling) | 6.02 / 14.5 / 18.1 | 11.5 |

Leading single levers (full 145-point runs): 6 cores 2.13 / 3.57 / 3.99, 200 MHz 3.43 / 3.69 / 3.91, vi1 1.91 / 2.35 / 2.41. Needed for 4x: 4 / 2.07 = 1.93x of hardware on top of the model's round 3 (2.17x if the low software case holds).
LINE_HALF is in no combination (k6 x 1.07 is about 4.1 central, about 3.5 low). Effects are sub-multiplicative (k6 is 1.86x against 1.96x from multiplying its singles).

## Costs (`lever_costs.txt`; no new synthesis; recorded vs derived marked per row)
Cores: a tile is 247k LUT in a 434k-LUT SLR; two tiles per SLR need a core at most about 40 % of an SLR, a shrink of at least 35 % (derived). Clock: recorded kernel WNS +0.111 ns at 10 ns; `doc/r42_arch_headroom.md` infers about
100-110 MHz; 125-150 MHz needs a pipelined LSU and replicated enable trees; 200+ MHz is a new physical design; no recorded screen above 100 MHz. LINE_HALF +597 LUT per core (recorded). vi1 not recorded, order of +14k LUT per core
if VRF read duplication (inference). Hardware loop +0.3-0.5k (derived). 2-wide front end +6-12k LUT-eq (derived). SFU slice at most about 17k (derived upper bound). Outlets 7.4k each, 1.5k with `AXI_LINE_LUTRAM` (recorded screen), all in SLR0
(about 4k LUT headroom). NUCA slice 4x in LUTRAM does not fit; URAM 0 of 960 used and needs new RTL.

## Ranked levers (predicted gain; confidence; cost class)
1. Clock: 1.79x at 200 MHz; model confidence high (wall time is arithmetic), feasibility low (recorded evidence supports about 100-110 MHz).
2. Cores 3 to 6: 1.68x; confidence medium (no 6-core board data); does not fit at today's tile size.
3. vi1: 1.13x; confidence medium-low (mechanism unverified); cost medium (VRF read path).
4. LINE_HALF: 1.07x; confidence high, RTL-checked; cheapest (+597 LUT per core).
5. 2-wide issue, hardware loop, zero-cost branch: 1.06x, 1.03x, 1.03x, overlapping upper bounds; low confidence; low to medium cost.
6. NUCA 4x: 1.05x; medium confidence; new slice RTL.
7. Tensor-engine slots, HBM latency, prefetch, SFU slices, word units: 1.01x-1.03x each.

## Statement on 4x
No single lever reaches 4x (200 MHz 3.69 central, 6 cores 3.57). Without new cores or clock closure the best set run at 150 MHz is 3.29 central (low 2.72). 200 MHz with 3 cores (k6): 3.85 central, low 3.38, about 4.1 with LINE_HALF;
k6 assumes extra pipeline registers on on-chip paths. 6 cores at 200 MHz: 5.6 to 7.3 central (low 4.14-4.25 for k1/k2/k8, 3.88 for k5); hardware-only (round-2 software) 4.2 to 5.5 central. 8x appears only in the unphysical ceiling (k9 10.8, k10 14.5).
Every combination at or above 4x needs either about 200 MHz or 6 cores, and neither is supported by recorded evidence.

## Deviations and time
About 6.2 h GB10 wall time at nice 19, at most 3 own processes at a time (one smoke test of about 3 minutes at 4); thor not used. Interactions are sub-multiplicative and LINE_HALF is added only as an analytic factor. Runs share the TNCA compile cache, which
moves one point by 2 % between fresh and cached runs (negligible for the geometric mean). Not validated against the board: xl composites (planning-share parts, analytic), large-M FP8 (wave model), proxies for 20 heavy points (validated against direct runs only for
silu, softmax and transpose, to 1 %), and all 6-core and 200 MHz predictions.
