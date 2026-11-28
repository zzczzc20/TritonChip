# R42 issue: results (worker `issue`, branch R42-issue, base R42 129f213d)

Domains in every number: **rtl-probe** (core testbench, assembly microbenchmarks, one core), **rtl-1c** (single-core RTL simulation of ArchBench points,
`archbench.run --counters`, profile `core-vrf-sync-v5-x8-g2c4-dp`), **rtl-3c** (three-core board-top RTL stand-in at the run-3 image geometry, `tools/r42/img_mc.py`),
**OOC** (Vivado 2023.2 out-of-context synthesis / place and route on xcu280, x86 host), **TLM** (the dse worker's SystemC model; its files were only read),
**derived** (arithmetic on recorded numbers). No board number exists for anything here. No link, card load or board access was made.

## 0. Result in one table, then one paragraph per finding

All effects below are hardware or software effects on the cycles of the run-3 image's machine at the same clock; "derived" = the img worker's group rule applied to measured ratios (`issue_predict.py`, `img_predict.py`), "model" = the dse worker's S145 central one-lever effect. **Direction of ratios in this table: rtl-1c and the probe are base/arm (above 1 = faster); rtl-3c are arm/base (below 1 = fewer cycles, 0.818x = 1.22x faster in the stand-in).** **Nothing reaches the 4x expectation: the levers built or measured here are worth about +5 % on the board geometric mean together.**

| lever (brief) | model (TLM) | built / measured | measured effect | derived board effect (central) | price (OOC tile) | verdict |
|---|--:|---|---|--:|---|---|
| (A) `vi1` single-cycle issue | +13 % | `VRF_FWD` (W-register forward, per-source early reads); step-0 probe | rtl-1c tiny 1.021x (27 pts), rtl-3c 0.981x (12 pts, inside noise), RTL bound of the whole mechanism 1.067x | +1.3 % (rtl-1c), +0.4 .. +0.8 % (rtl-3c) | +0.9 k LUT, `r_fwd` -0.69 ns at 5.6 ns (not the worst path) | keep; not worth a link alone |
| (A') bank-aware allocation (existing knob `TNCA_BANK_ALLOC=1`) | +0.2 % (`vrf0`) | measured only | rtl-1c tiny 1.019x (27 pts) | +1.1 % | none (software) | recommend default after the gates |
| (B) `sets128` | +4.7 % | image parameters wired, stand-in `cache_sets` 128, `M_SCAN` priced | rtl-3c 0.938x geometric mean (17 pts; small tier 0.67-0.96) | +1.8 % (113 pts) | **+11.5 k LUT; NUCA-slice cone -0.4 ns at 5.6 ns** | the largest of the three, pays in timing |
| (C) `te32` | +3 % | `TENSOR_TAG_W 5` (te16) measured, te32 not buildable | rtl-3c te5 alone 0.974x; with LF 1.000x | +0.9 % alone, 0 with LF | +4.5 k LUT | not worth |
| (D) `hl` | +3 % | not built | ceiling 1.9 % of cycles | <= 1.9 % | - | not built |
| LF + VRF_FWD + sets128 (candidate) | x1.27 (product of single levers) | image `issue_cand.tcl`, profile `core-r42-issue` | rtl-3c 0.818x (15 pts, outputs bit-identical) | +5.3 % (113 pts), +4.4 % (145 pts) | +13.4 k LUT, WNS -1.90 vs -1.52 at 5.6 ns | does not meet the lead's trigger (+25 %) |

1. **The 2-cycle-per-vector-instruction mechanism of `vi1` does not exist in the run-3 core.** With `VRF_PREFETCH 1` + `VRF_BYPASS 1` (the image's parameters) independent vector
   instructions of 20 kinds (0-3 VRF sources, int/FP/convert/shuffle) issue at exactly 1.00 cycle each; 2.00 appears only with `VRF_PREFETCH 0`. What costs cycles is
   (a) a dependent consumer: producer in W, consumer reads the bank in R (request cycle + response cycle), (b) two sources in one VRF bank (+1 cycle per extra source), (c) refill after a taken branch. The TLM's `issue_vector_cycles 2` is a calibration knob, not a mechanism: its
   tiny-tier `vi1` effect is 1.165x, the RTL bound for removing every operand-fetch stall cycle is 1.067x (geometric mean over the same 27 points).
2. **`VRF_FWD` (built, default off) is the cheapest change that overlaps the operand fetch of a dependent instruction**: the writeback register is forwarded into R's operand capture; the other sources of a waiting instruction are read meanwhile. Measured (rtl-1c tiny, 27 points): 1.021x geometric mean (1.025x cycle-weighted).
3. **The larger issue-side loss is bank conflicts, and the compiler already has the fix, off by default (`TNCA_BANK_ALLOC=1`)**: 1.019x on the 27 tiny points; together with VRF_FWD 1.039x (1.050x cycle-weighted), `x_vrf` stall cycles -67 .. -82 % on the model points.
4. **`hl` / zero-cost branch: not built.** Ceiling 1.9 % of cycles (6,376 redirects x 4 cycles over 1.34 M cycles of the 27 tiny points); a predictor needs a front-end change in the path `clk` is re-pipelining; a hardware loop is a new encoding.
5. **`te32` does not fit this core as modelled** (needs 128 LSU line slots); `te16` helps only without LINE_HALF.
6. **`sets128`**: image parameters only; the stand-in refused `cache_sets` above 64 and had no `SHRINK` parameter (fixed). No benchmark point issues a cache-maintenance operation, so the 2,048-cycle `M_SCAN` costs 0 cycles in every measured launch; the board-top image geometry test (which does use the handoff) passes for the candidate.

(Sections 1-8 carry the numbers, section 9 what failed or was not done, section 10 the ranked ideas.)

## 1. Step 0: the probe (rtl-probe; `tools/r42/issue/probe/`, tables `probe/*.tsv`)

Method: `gen_probe.py` writes straight-line assembly streams (160 instructions after a short preamble) of one instruction kind; `probe_run.py` runs them in `tb/core/tb_tnca_core.v`
(run-2 machine of `make core-run2`: banked VRF, `VRF_PREFETCH 1`, `VRF_BYPASS 1`, `FE_NOBUBBLE 1`, `RCTX_SWITCH 2`, one context) with `+trace` (one line per instruction in X) and reports
the gap between consecutive instructions. Variants: `run2` (the run-3 core), `nopf` (`VRF_PREFETCH 0`), `nobyp` (`VRF_BYPASS 0`), `behav` (behavioural register file, read in X with forwarding: the bound of every VRF read lever),
`fwd` (`VRF_FWD 1`). A first run with `+allroles` (four contexts interleaved) was a mistake and is not used (`probe1.tsv` not kept); the one-context tables are `probe_r1.tsv`, `probe_r4.tsv`.
`tb_probe_core.v` is the bench plus a per-cycle pipeline monitor (`+pipetrace`), testbench only.

**Independent streams, cycles per instruction** (`probe_r1.tsv`): run-3 core 1.00 for every kind: `s.add`, `s.movz`, `v.movi`, `v.iota`, `v.splat`, `v.addi`, `v.shli`, `v.add`, `v.mul`, `v.and`, `vf.add`, `vf.mul`, `vf.neg`, `vf.fma` (3 VRF sources),
`vf.cvt.s32.f32`, `v.rotate`, and alternating mixes (134 of 134 gaps = 1 in every program). `VRF_PREFETCH 0`: 2.00 for every kind that reads the VRF (1.50 in the alternating mixes), 1.00 for `v.movi`/`v.iota`/`v.splat`/scalar.
`VRF_BYPASS 0` does not matter for independent streams. With four contexts of one program (`probe_roles4.tsv`) the aggregate rate of the run-3 core is 1.0 instruction/cycle (4 cycles per instruction per context) for independent and
for chained programs alike; `VRF_PREFETCH 0` gives 0.5.

**Dependent instructions** (`probe_r4.tsv`; cycles per instruction of the stream):

| program | run-3 core | VRF_FWD | behavioural VRF |
|---|--:|--:|--:|
| `dist1_int` (producer, consumer of its result into another register, repeat) | 2.0 | 1.5 | 1.0 |
| `v2_chain1` `v.add v4,v4,v2` (RAW and WAW on one register) | 3 | 2 | 2 |
| `v1_chain1` `v.addi v4,v4,#1` | 3 | 2 | 2 |
| `f2_chain1` `vf.add v4,v4,v2` | 5 | 4 | 4 |
| `f3_chain1` `vf.fma v4,v1,v2,v4` | 5 | 4 | 4 |
| `v2_chain2` (two interleaved chains) | 1.5 | 1.0 | 1.0 |
| `f3_chain2` | 3 | 2 | 2 |
| `v2_add_bank_conflict` `v.add vd,v1,v5` (v1 and v5 in bank 1) | 2 | 2 | 1 |
| `f3_fma_bank_conflict2` (two sources in one bank) | 2 | 2 | 1 |
| `f3_fma_bank_conflict3` (three sources in one bank) | 3 | 3 | 1 |

Cause (from `+pipetrace`, `probe` notes): the producer is in X at t, in W at t+1 (integer; FP: t+3); the consumer is hazarded in R until the scoreboard shows the write (`r_whit`), then issues its bank reads in
that cycle (the bank's write-first path returns the new value) and leaves R one cycle later: consumer in X at t+3. `VRF_FWD` takes the value from the W register in the cycle it is written: consumer in X at t+2. The behavioural file reads in X and
forwards the W write, i.e. at t+1 for a consumer that does not rewrite the register. Sources in one bank are read one after the other (one read port per bank).

**Counted loop** (`probe_loops.tsv`): `K` independent vector instructions + `s.subi` + `br.nz`, cycles per iteration: 6 for `K = 0`, K+6 for the integer body (7, 8, 10, 14 for K = 1, 2, 4, 8); the taken branch and the scalar RAW cost
about 4 cycles per iteration beyond the issue slots, independent of the register file variant.

**Real kernels** (rtl-1c, 27 tiny points, `rtl_1c/tiny_base.jsonl.gz`, table `probe/cnt_tiny_base_table.tsv`): exclusive stall counters as fractions of cycles, cycle-weighted over the 27 points (1.34 M cycles):
issued 0.449, LSU 0.127, transfer engine 0.106, hazard 0.088, **operand fetch (`x_vrf`, the second R cycle of a non-prefetched read) 0.074**, matrix 0.041, frontend 0.043, barrier sleep 0.031, FPU 0.024, reservation/steal 0.020.
Removing all `x_vrf` cycles bounds the per-point geometric mean at **1.067x** (cycle-weighted 1.080x); `x_vrf` + `hazard` together 1.143x (hazard includes load-use waits, not removable by forwarding).
Per-point TLM `vi1` against this bound: `tlm_vi1_vs_rtl_bound.txt` (TLM 1.165x geometric mean, RTL bound 1.067x, same 27 points).
Behavioural register file on the model points that finished: gpt2_block 1.088x, llama_decode 1.039x, llama_prefill 1.045x, mlp 1.119x (`rtl_1c/tiny_behav.jsonl.gz`); `VRF_PREFETCH 0`: 0.904, 0.900, 0.830, 0.950 (`tiny_nopf.jsonl.gz`;
the other points of that run are runtime faults, see section 9).

## 2. What was built and changed (all default-off; every default keeps the shipped RTL bit- and cycle-identical, see section 3)

`VRF_FWD` (tnca_core parameter, default 0; image parameter `VRF_FWD`):
- `rtl/core/tnca_core.v`: (1) `w_full` (all sixteen lanes of the W write enabled, registered with `w_lane_en`) and `r_fwd[2:0]` (R's source k is the register W writes this cycle with all lanes; next-state of `r_whit` ANDed with the lane
  test, registered); (2) a forwarded source counts as arrived (`of_arrive = of_arrive_rsp | of_fwd`, so `of_have`, `of_ready`, `of_want`, `r_left` and D's prefetch gate follow without further change), its data is `w_vdata`
  (`of_v* = of_fwd ? w_vdata : of_arrive_rsp ? bank response : of_d*`), it is not held (the holding registers keep their single input; if the instruction cannot leave R that cycle the source is read from the bank in the next one);
  (3) per-source operand fetch: with `VRF_FWD` a source without a producer in flight is read while another source of the same instruction waits (`of_need` per source from `r_vhz`; the old rule fetched nothing until no producer of any source was in flight),
  and D prefetches its hazard-free sources while another source of D waits (`pf_issue = pf_need & ~pf_hz_v`); (4) `r_of_idle` (the RCTX_SWITCH give-way test) ignores held data with `VRF_FWD`, so an instruction that waits with early-read sources can still give way.
  A partial-lane write is not forwarded (the bank merges it with the old value); the instruction reads the bank as before.
- Parameter threading, same pattern as `MTX_PIPE`: `rtl/tile/tnca_tile.v`, `rtl/tnca_top.v`, `rtl/tnca_multicore_shell.v`, `rtl/tnca_shell.v` (+ `rtl/common/tnca_uarch_defaults.vh`), `rtl/board/*`, `rtl/fpga/*`, `tb/core/tb_tnca_core.v`, `tb/core/tb_core_hybrid.v`,
  `tb/device/*`, `tb/fpga/tnca_multicore_board_regs.v` (also gains `SHRINK`, which the stand-in bench lacked), `python/tnca/device.py` (`VRF_FWD` in the RTL switch list: `TNCA_RTL_PARAMS=VRF_FWD=1`), `python/tnca/backends/multicore_file.py`,
  `tools/r41/link_budget_rtl.py`, `tools/r40/image_geometry_tests.py`, `python/tnca/backends/multicore_mmio_sim.py` (`cache_sets` may be 128 and 256: it refused anything above 64).
- Makefile: `R42_VRF_FWD ?= 0` on the run-2 core machine (`make BUILD=sim/build_fwd core-run2 R42_VRF_FWD=1`).
- Profiles `core-r42-fwd` (run-3 core + `VRF_FWD 1`) and `core-r42-issue` (`core-r42-img-lf` core + `VRF_FWD 1`; the single-core device of the combination image) in `spec/r1_profiles.json`, targets `compiler/targets/tnca-sim-r42-{fwd,issue}.json`
  and `doc/support_matrix.md` regenerated with the repo generators; both execute the artifacts of `core-vrf-sync-v5-x8-g2c4-dp` unchanged. `make r1-profile-tests`: `core-r42-fwd` 10 facts read back PASS (the overall CFG-01 line fails in this tree
  only for the other R42 profiles whose benches (`tb_core_r42*.vvp`) are not built here, as before this branch).
- Tools (`tools/r42/issue/`): `probe/` (generator, runner, loop probe, pipeline monitor bench), `lockstep_gen.py` / `lockstep_run.py` (random programs, memory-image and cycle comparison of two builds, 1/2/4 contexts), `cnt_table.py`, `cnt_cmp.py`, `gate_cmp.py`,
  `gates_core.sh`, `verify_v2.sh`, `iss_mc.sh` (stand-in arms), `mc_table.py`, `issue_predict.py` (145-point derived effect), `iss_queue.sh` / `iss_job.sh` / `iss_tilejob.sh` + `queue_*.txt` (x86 OOC queue with memory gate and stop file), `ooc_collect.py`, `ep_compare.py`,
  `stage.sh`; `tools/r42/issue_images.py` (image parameter files of every arm, `tools/r42/issue/images/*.tcl`).
- Not changed: the compiler, the ISA, any default, any shipped parameter.

## 3. Verification of VRF_FWD (rtl; gates run on the final RTL, `gates/`, `lockstep/`)

**Default off is cycle-identical.** `make core-run2` on the base tree (R42 129f213d sources) against this tree at defaults: 88 program runs each (44 programs, in-order and reordered memory), all PASS,
**88/88 identical cycle counts** (`gates/core_run2_compare.txt`). Also at defaults on this tree: `make core-g2c4` exit 0, `make core-mmadp` exit 0, `make unit` exit 0 (11 unit benches + asm round trip 56,650 instructions).
On real kernels: rtl-1c gpt2_block 173,772 and llama_decode 177,091 cycles are identical between the tree before and after the parameter plumbing (`rtl_1c/tiny_base2.jsonl.gz` against `tiny_base.jsonl.gz`).
**Results are unchanged with the parameter on.** `make BUILD=sim/build_fwd core-run2 R42_VRF_FWD=1`: 88 runs PASS (self-checking programs; cycles geometric mean 0.993 of the default, 47 of 88 differ).
Lockstep (`lockstep_gen.py`, `lockstep_run.py`): random programs (integer, FP, convert, SFU, rotate, reductions to scalar, lane-masked and predicated forms, loads, stores, short counted loops, dependent chains of random distance) run on the run-3 core and on the
same core with `VRF_FWD 1`, the memory image written by the program compared word for word, with 1, 2 and 4 contexts running the same program:

| RTL | programs x contexts | runs | image mismatches | cycles cand/ref per context count (slower runs / runs) |
|---|---|--:|--:|---|
| v1 (first version; the holding registers took the forwarded data) | 40 x 3 | 120 | 0 | 1 ctx 0.944 (0/40), 2 ctx 0.974 (7/40), 4 ctx 0.994 (12/40) |
| v1 | 100 x 3 | 300 | 0 | 1 ctx 0.945 (0/100), 2 ctx 0.978 (14/100), 4 ctx 0.998 (34/100) |
| **v2 (final)** | 100 x 3 | 300 | **0** | 1 ctx 0.950 (0/100), 2 ctx 0.979 (18/100, worst +6.9 %), 4 ctx 0.998 (33/100, worst +3.9 %) |

(v1 and v2 differ only in where a forwarded value lives; v2 is the committed RTL and the one every other number in this file uses, except the v1 tile rows marked as such in section 6.) With four contexts the interleaving already hides the dependent latency,
so the forward changes the aggregate by 0.2 %; a minority of runs is slower by up to 7 % because the instruction order of the contexts changes (give-way and refetch timing), the memory image is identical in all 720 runs.
Sensitivity of the lockstep (`tools/r42/issue/mutants.sh`, `lockstep/mutants.log`): three single-point mutants of the forward, 30 programs x 1 and 2 contexts each, against the unmutated `VRF_FWD 0` core:
`wn_full` forced to 1 (partial-lane writes are forwarded): 53 of 60 runs differ in the memory image; the select of source c taken from source b's forward flag: 28 of 60 differ; a forwarded source held in `of_got` without data (the v1 bookkeeping on the v2 data path): 43 of 60 differ.
The unmutated RTL differs in none of 720 runs, so the comparison does see a wrong forward.
## 4. Real kernels on the single-core RTL (rtl-1c, ArchBench tiny tier, run-3 core profile `core-vrf-sync-v5-x8-g2c4-dp`, `--counters`)

Device cycles of the timed call; ratio = base cycles / arm cycles (>1: faster). All 27 points pass in every arm and `max_rel_err` is identical per point between the base and VRF_FWD (hardware change, same software).
`VRF_FWD` through `TNCA_RTL_PARAMS=VRF_FWD=1`; bank-aware allocation through `TNCA_BANK_ALLOC=1` (software knob, base hardware); both together. Records: `rtl_1c/tiny_{base,fwd2,bankalloc,fwdbank}.jsonl.gz`.

| point | base cycles | VRF_FWD | bank alloc | VRF_FWD + bank alloc | x_vrf base -> fwd+bank |
|---|--:|--:|--:|--:|--:|
| model.gpt2_block/tiny/fp16-d64-h4-b1-s16 | 173772 | 1.045 | 1.020 | 1.068 | 14779 -> 4840 |
| model.llama_decode/tiny/fp16-b1-p16 | 177091 | 1.013 | 1.020 | 1.026 | 7218 -> 1707 |
| model.llama_prefill/tiny/fp16-b1-s16 | 473768 | 1.017 | 1.021 | 1.027 | 27845 -> 7675 |
| model.mlp/tiny/fp16-t16-d64 | 79981 | 1.070 | 1.014 | 1.092 | 7238 -> 2144 |
| model.resnet_block/tiny/fp32-n1-c8-hw8 | 246856 | 1.024 | 1.042 | 1.068 | 25055 -> 4454 |
| op.add/tiny/fp16-n1000 | 2373 | 1.003 | 1.012 | 1.015 | 67 -> 8 |
| op.add/tiny/fp32-n256 | 828 | 1.006 | 1.000 | 1.006 | 1 -> 1 |
| op.argmax/tiny/fp16-r2-v256 | 1999 | 1.003 | 1.023 | 1.017 | 213 -> 150 |
| op.attention_decode/tiny/fp16-b1-h2-l16-d16 | 9188 | 1.058 | 1.030 | 1.096 | 711 -> 154 |
| op.attention_prefill/tiny/fp16-b1-h2-s16-d16 | 23340 | 1.035 | 1.026 | 1.060 | 2574 -> 967 |
| op.bmm/tiny/fp16-b2-m16-k16-n16 | 1858 | 1.018 | 1.004 | 1.016 | 46 -> 10 |
| op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8 | 110375 | 1.022 | 1.055 | 1.102 | 11700 -> 2118 |
| op.cumsum/tiny/fp32-r4-c64 | 1078 | 1.000 | 1.043 | 1.043 | 44 -> 0 |
| op.embedding/tiny/fp16-v256-d64-t16 | 4251 | 1.000 | 1.021 | 1.021 | 96 -> 0 |
| op.gelu_tanh/tiny/fp16-n256 | 1971 | 1.002 | 1.034 | 1.024 | 164 -> 98 |
| op.gemv/tiny/fp16-k64-n64 | 4763 | 1.012 | 1.000 | 1.012 | 33 -> 3 |
| op.layernorm/tiny/fp16-r4-d64 | 2046 | 1.036 | 1.041 | 1.083 | 116 -> 53 |
| op.linear/tiny/fp16-m16-k64-n64 | 5577 | 1.001 | 0.997 | 1.003 | 22 -> 4 |
| op.matmul/tiny/fp16-m16-k32-n16 | 1009 | 1.000 | 1.000 | 1.000 | 0 -> 0 |
| op.matmul/tiny/fp32-m16-k32-n16 | 3182 | 1.018 | 1.015 | 1.033 | 470 -> 350 |
| op.rmsnorm/tiny/fp16-r4-d64 | 1409 | 1.150 | 1.022 | 1.158 | 67 -> 30 |
| op.rope/tiny/fp16-b1-h2-s16-d16 | 3872 | 1.003 | 1.043 | 1.046 | 182 -> 0 |
| op.silu/tiny/fp16-n256 | 1460 | 1.025 | 1.008 | 1.030 | 124 -> 79 |
| op.softmax/tiny/fp16-r4-c64 | 2467 | 1.004 | 0.952 | 0.995 | 223 -> 73 |
| op.softmax/tiny/fp32-r3-c37 | 1343 | 1.000 | 1.037 | 0.989 | 123 -> 44 |
| op.swiglu/tiny/fp16-n256 | 1659 | 0.999 | 1.018 | 1.014 | 131 -> 85 |
| op.transpose/tiny/fp32-r32-c32 | 2274 | 1.012 | 1.034 | 1.039 | 118 -> 16 |
| **geometric mean (27)** | | **1.021** | **1.019** | **1.039** | |
| cycle-weighted | 1339790 | 1.025 | 1.027 | 1.050 | |

Reading: the forward is worth most where chains of dependent vector instructions meet few contexts (rmsnorm 1.15, mlp 1.07, attention_decode 1.06, gpt2_block 1.045) and nothing where the kernel is matrix- or transfer-bound (matmul/linear/gemv tiny 1.00-1.01).
Bank-aware allocation gains on almost every point (1.00-1.055) and loses on two (softmax/fp16 0.952, linear 0.997). The two are close to additive (1.021 x 1.019 = 1.040 against 1.039 measured). Together they remove 67-82 % of the `x_vrf` stall cycles on the model points
(gpt2_block 14,779 -> 4,840, resnet_block 25,055 -> 4,454); the behavioural single-cycle register file bound on the four model points is 1.04-1.12x.
Derived effect on the 145-point board geometric mean (`issue_predict.py`, `predict_issue_side.txt`; group-by-(family, dtype) rule of `tools/r42/img_predict.py`, 27 measured groups, unmeasured families 1):

| effect | measured points only | every board point of a measured group | same, log halved for medium/large/hbm/xl (central) |
|---|--:|--:|--:|
| VRF_FWD | +0.38 % | +1.87 % | **+1.32 %** |
| bank-aware allocation | +0.36 % | +1.56 % | **+1.12 %** |
| both | +0.72 % | +3.48 % | **+2.46 %** |

The dse model's `vi1` is +13 % (central) for the same mechanism. Model against RTL on the tiny tier: TLM 1.165x, RTL VRF_FWD 1.021x, RTL bound (all `x_vrf` removed) 1.067x, RTL both 1.039x. The TLM's calibration made `issue_vector_cycles 2` stand in for what the RTL spends
on operand fetch, dependences, bank conflicts and refill together; it also gives 2 cycles per vector instruction regardless of the number of contexts, while the RTL issues 1.0 per cycle in aggregate with 1 to 4 contexts (probe, `probe_roles*.tsv`).

## 5. Three-core RTL stand-in (rtl-3c: `tnca_multicore_board_regs` at the run-3 image geometry; GB10 and thor)

Method: `tools/r42/img_mc.py` (the img worker's wrapper of `tools/r41/link_budget_rtl.py`: the point runs exactly as on the board, `torch.compile` on device `tnca`, default knobs, a cold call then the timed call), image parameter files `tools/r42/issue/images/issue_<arm>_sim.tcl`
(the run-3 parameters plus the arm's: `fwd` = `VRF_FWD 1`; `s128` = `SETS 128, SET_W 7, SHRINK 10`; `te5` = `TENSOR_TAG_W 5`; `lf_fwd_s128` = `TENSOR_LINE_HALF 1, MAC_F32 1, VRF_FWD 1, SETS 128, SET_W 7, SHRINK 42`; `lf_fwd_s128_te5` adds `TENSOR_TAG_W 5`).
**Every arm is compiled under the same profile name (`core-vrf-sync-v5-x8-g2c4-dp`), so the artifacts are those of the base arm** (the img worker's name effect, +-3-9 % on pointwise tiny points, does not enter). 17 points: the tiny points of matmul fp16/fp32, linear, gemv, mlp, bmm, attention_decode, softmax, rmsnorm,
conv2d, add, the small points of matmul, linear, gemv, softmax, rmsnorm and the FP8 Hub proxy `op.fp8_linear/xl/fp8-m16-k256-n128`. Device cycles of the timed call; ratio = arm / base (<1: fewer cycles); `=` output sha256 equal to the base's (every arm record: `=`), records in `rtl_3c/*.tgz`.
Columns: `base` = the first run of the run-3 image (thor, cold compile cache; the FP8 proxy point faulted in that run, `DEVICE_LOST ... invalid or unknown value`, kept in `thor_base.tgz`; the re-run, 61,835 cycles, is used), `thor_base_v2` = the same image re-run later from the final tree (default parameters), `gb_s128`, `thor_fwd`, `gb_te5`,
`combo_all` = LF + VRF_FWD + sets128 (tiny: GB10 `gb_combo_T`, small: `gb_combo_S`), `thor_cand` = the same plus `TENSOR_TAG_W 5` (tiny only).

```
point                                                base    thor_base_v2         gb_s128        thor_fwd          gb_te5       combo_all       thor_cand
model.mlp/tiny/fp16-t16-d64                         44046               -    40354 0.916=    42538 0.966=    41369 0.939=    33800 0.767=    33810 0.768=
op.add/tiny/fp32-n256                                 591               -      591 1.000=      594 1.005=      591 1.000=      593 1.003=      593 1.003=
op.attention_decode/tiny/fp16-b1-h2-l16-d16          6495               -     6419 0.988=     6372 0.981=     6417 0.988=     6447 0.993=     6375 0.982=
op.bmm/tiny/fp16-b2-m16-k16-n16                      1590     1586 0.997=     1590 1.000=     1572 0.989=     1535 0.965=     1380 0.868=     1381 0.869=
op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8             58670               -    58942 1.005=    58846 1.003=    58933 1.004=               -               -
op.fp8_linear/xl/fp8-m16-k256-n128                  61835               -    60911 0.985=    55862 0.903=    61560 0.996=               -               -
op.gemv/small/fp16-k256-n512                        70175               -    62751 0.894=               -               -    46980 0.669=               -
op.gemv/tiny/fp16-k64-n64                            4811     4814 1.001=     4822 1.002=     4775 0.993=     4758 0.989=     3463 0.720=     3469 0.721=
op.linear/small/fp16-m64-k256-n256                 169131               -   112968 0.668=               -               -    92219 0.545=               -
op.linear/tiny/fp16-m16-k64-n64                      4451               -     4441 0.998=     4449 1.000=     4078 0.916=     3730 0.838=     3717 0.835=
op.matmul/small/fp16-m64-k128-n128                  40894               -    31828 0.778=               -               -    23714 0.580=               -
op.matmul/tiny/fp16-m16-k32-n16                      1908     1910 1.001=     1909 1.001=     1909 1.001=     1720 0.901=     1558 0.817=     1560 0.818=
op.matmul/tiny/fp32-m16-k32-n16                      3256               -     3257 1.000=     3201 0.983=     3257 1.000=     3201 0.983=     3202 0.983=
op.rmsnorm/small/fp16-r64-d256                      18430               -    15729 0.853=               -               -    15325 0.832=               -
op.rmsnorm/tiny/fp16-r4-d64                          1454     1432 0.985=     1432 0.985=     1393 0.958=     1453 0.999=     1388 0.955=     1394 0.959=
op.softmax/small/fp16-r64-c512                      93287               -    89908 0.964=               -               -    85284 0.914=               -
op.softmax/tiny/fp16-r4-c64                          1812               -     1809 0.998=     1806 0.997=     1807 0.997=     1806 0.997=     1808 0.998=
geomean arm/base (n)                                          0.996 ( 4)     0.938 (17)     0.981 (12)     0.974 (12)     0.818 (15)     0.888 (10)
'=' output digest identical to the base's, '!' differs or missing
```

**Noise of this stand-in:** the same image and software re-run (`thor_base_v2` against `base`) differs by 0.1-0.3 % on the GEMM-shaped points and by -1.5 % on `rmsnorm/tiny` (1,454 -> 1,432; the first run of a point compiles into a cold cache, later runs reuse the compiled artifact, the dse worker saw the same cache effect on
`op.transpose/tiny`). Effects under 2 % on pointwise tiny points are therefore not resolved here; rtl-1c (section 4) is deterministic and carries the `VRF_FWD` number.
Reading: (1) **sets128** is where the stand-in sees capacity: linear/small 0.668, matmul/small 0.778, gemv/small 0.894, rmsnorm/small 0.853, mlp/tiny 0.916, softmax/small 0.964, the tiny GEMM points 1.00; geometric mean 0.938 over the 17 points; digests equal.
(2) **VRF_FWD** on the stand-in: 0.981 over 12 points (rmsnorm/tiny 0.958 and the FP8 proxy 0.903 stand out, GEMM points 0.98-1.00): inside the noise except the FP8 proxy and rmsnorm. (3) **te5** alone: matmul/tiny 0.901, linear/tiny 0.916, mlp/tiny 0.966, geometric mean 0.974;
**with LF on it adds nothing**: `combo_all` and `thor_cand` agree to 0.3 % on every point they share (matmul/tiny 0.817/0.818, bmm 0.868/0.869, gemv 0.720/0.721, linear 0.838/0.835, mlp 0.767/0.768): the line engine takes the rows that the wider word-walk would have helped, as the lineelig engine bench said.
(4) **The combination** LF + VRF_FWD + sets128: geometric mean 0.818 over its 15 points (matmul/small 0.580, linear/small 0.545, gemv/small 0.669, gemv/tiny 0.720, mlp/tiny 0.767, bmm 0.868, rmsnorm/small 0.832, softmax/small 0.914; conv2d/tiny 1.004 and add 1.003 untouched).
For comparison the img worker's LF arm against the same image (`evidence/r42/img/rtl_3c_tables.txt`, under a different profile name): matmul/small 0.584, gemv/small 0.799, matmul/tiny 0.818, gemv/tiny 0.719, linear/tiny 0.837, bmm 0.875: on the tiny GEMM points the combination equals LF; on matmul/small the capacity gain is already inside LF's
(0.580 against 0.584), on gemv/small it adds 0.799 -> 0.669.

**Derived effect on the board geometric mean** (`predict_standin.txt` with `issue_predict.py`, the 145-point rule; `predict_standin_img_predict113.txt` with the img worker's own `img_predict.py` on the 113 points of ab-int3, base 2.123x; groups only where measured, unmeasured families 1):

| arm | 113 points, measured only | family | family/2 (central) | `models` | 145 points, family/2 |
|---|--:|--:|--:|--:|--:|
| sets128 | +1.0 % | +2.5 % | **+1.8 %** | +2.6 % | +1.6 % |
| VRF_FWD (stand-in) | +0.1 % | +0.5 % | +0.4 % | +0.7 % | +0.8 % |
| `TENSOR_TAG_W 5` | +0.3 % | +1.3 % | +0.9 % | +1.5 % | +0.8 % |
| LF + VRF_FWD + sets128 | +2.7 % | +7.5 % | **+5.3 %** | +7.8 % | +4.4 % |

The img worker's LF alone: +5.9 % (`family/2`), +7.9 % (`models`) on the same 113 points with more measured families, so the combination is not resolved above LF by this point set; the increments of sets128 and VRF_FWD over LF are at most about +1-2 % each.
The model (`R42-dse`) says +4.7 % (sets128), +13 % (vi1), +7 % (le2); multiplied as single levers (not a combined TLM run) 1.047 x 1.13 x 1.07 = 1.27 against the +5 .. +8 % derived here.

## 6. Area and timing (OOC; x86 host; `ooc/` holds the reports, `ooc_summary.txt` the table)

Every row is the same `tnca_tile` (tile 0 of a three-tile grid, run-3 parameters plus the **step-3 shrink parameters `FMA_IMPL 465` and `SHRINK 42`**, hierarchy kept, `tools/r42/issue/queue_v5.txt` has the exact generics) synthesised at 10 ns with `SYNTH_ONLY`, and then
placed and routed alone in SLR0 (`tools/r42/area/tile_screen.tcl`: no shell, no other tile, no SLR crossing, optimistic like every OOC tile screen; one run per cell, placement noise about +-0.4 ns). `ctl` is the identical-function control: the same tree with every parameter of this
study at its default (its tile is the shrink worker's step 3, 230,979 LUT in his tree, 229,400 here: Vivado is not stable at +-1.5k).

| tile | what | synthesis LUT (delta to ctl) | synthesis FF | routed WNS 10 ns | routed WNS 6.67 ns | routed WNS 5.6 ns (failing endpoints of ~195 k) |
|---|---|--:|--:|--:|--:|--:|
| ctl | control | 229,400 | 69,313 | +1.008 | -0.161 (1,134) | -1.522 (28,245) |
| fwd v1 | VRF_FWD, first version | 231,441 (+2,041) | 69,364 | +0.588 | - | stopped at 14 min (RTL replaced) |
| **fwd2** | **VRF_FWD final** | **230,287 (+887)** | 70,857 (+1,544) | +0.777 | not finished | -1.107 (18,646) |
| te5 | `TENSOR_TAG_W 5` | 233,943 (+4,543) | 70,432 | - | - | - |
| s128 | `SETS 128` + `SET_W 7` | 240,899 (+11,499) | 72,031 | - | - | - |
| lf | `TENSOR_LINE_HALF 1` + `MAC_F32 1` | 231,141 (+1,741) | 69,541 | - | - | - |
| **comb** | LF + VRF_FWD + sets128 (the candidate) | **242,821 (+13,421)** | 73,811 (+4,498) | +0.581 | not run | -1.900 (30,637) |
| comb5 | comb + `TENSOR_TAG_W 5` | 245,008 (+15,608) | 74,900 | - | - | unfinished (see below) |

Where the lever's own logic is (`ooc/*/synth_util_hier.rpt.gz`): the whole `VRF_FWD` increment is in the core's own logic (`(u_core)` 8,514 -> 10,574 LUT in v1, +2,060; v2 +887 LUT and +1,544 FF because the operand-capture registers `x_v?_q` can no longer be merged with the holding registers). No other unit moves.

**Timing at 5.6 ns (the target of the `clk` worker's OOC tile).** The tile alone is not at 5.6 ns today: ctl -1.522 (28,245 failing endpoints), at 6.67 ns -0.161 (1,134), at 10 ns +1.008; the worst paths are the hybrid LSU, the NUCA slice, the core's own logic, the vector FPU, the VRF
(ranked in `ooc/*/worst300_endpoints.csv.gz`). The lever's own endpoints, `endpoints.csv` of the 5.6 ns screens (`ep_compare.py`):
- `VRF_FWD`: the new register `r_fwd` (next state of `r_whit` ANDed with the all-lanes test, through `stall_front`, 18 levels, start `x_sd_q`) is a failing endpoint at 5.6 ns: **-0.69 ns** (fwd2 tile) and -0.67 ns (comb tile); the control's `r_whit` endpoints are at -0.41 (ctl) and -0.25 (fwd2).
  It is better than the tile's worst (-1.1 .. -1.5) so it is not the limiter, and the front-end/stall group (649 endpoints, worst -1.087 in ctl, -1.032 in fwd2) did not get worse. It belongs to the family `clk` is already pipelining (stall cone, `stall_front` fan-out); a version with `r_fwd = r_whit & w_full` taken at the point of use
  would take `stall_front` out of the new register's input, at the price of one more input in the operand-ready cone; not built.
- `sets128`: the home slice becomes the worst family at 5.6 ns: failing endpoints in the slice 851 (ctl) -> 11,202 (comb), worst slack -1.512 -> **-1.900** (`victim_ptr -> s3_wb`, 20 levels; `wbq_addr -> ms_state CE`, 28 levels; `s1_addr -> d_bits/v_bits`): the victim selection, the
  valid/dirty vectors and the write-back queue are 4x larger. At 10 ns the slice has 900 endpoints within 1.5 ns (worst +1.014 against none in ctl). The shrink worker's unit-level result (slice tag read -0.078 at 6.67 ns) is not this; the whole-slice cone at 128 sets is deeper than at 32.
  This is the lever's timing cost: **about 0.4 ns at 5.6 ns on the worst path of the tile** (comb -1.900 against ctl -1.522; fwd2 alone does not move the worst path).
- `TENSOR_TAG_W 5`: +4.5 k LUT (te5 against ctl) and nothing in the line-engine arms (section 5); excluded from the candidate.

Routed LUT of the screens at 10 ns: ctl 227,241, fwd2 228,147, comb 240,543. **SLR packing (derived):** the shrink worker's floorplan arithmetic puts SLR0 at 89.9 % with the step-3 tile; the candidate adds +13.4 k synthesis LUT per tile = about +13.2 k routed = +2.26 k CLB at 5.84 LUT/CLB of 54.3 k CLB
per SLR: +4.2 points, SLR0 about 94.1 % (highest packing that ever routed 96.74 %), SLR1 88.8 %, SLR2 85.5 %.
Not finished when this file was last edited (x86 jobs of mine still running; `comb5` at 5.6 ns was in post-route physical optimisation after 63 minutes, its log showed an interim path-group WNS of -2.85 ns inside `phys_opt`, which is not a routed result and is not used anywhere; the queue stop file `~/r42_issue_out/STOP` is set so no further job of mine starts; results land in `~/r42_issue_out/<name>` and are summarised by `python3 tools/r42/issue/ooc_collect.py ~/r42_issue_out`): `tscr_i_s3_comb5_p56`, `tscr_i_s3_fwd2_p667`; not run: `tscr_i_s3_comb_p667`, the 5.6 ns screens of te5, s128 and lf alone, the core-unit runs (`queue_core.txt`).

## 7. The other levers of the brief

**(B) `sets128` and the maintenance scan.** The parameters exist: image `SETS 128`, `SET_W 7`, `SHRINK` (bit field of the shrink worker: DATA_RAM bits 1:0, AUX_RAM bits 3:2, VRF_XBAR bits 5:4; 10 = both arrays in URAM, 42 = and the VRF response crossbar).
Wiring added here: the stand-in refused `cache_sets` above 64 (`multicore_mmio_sim.py`) and its bench had no `SHRINK` (`tnca_multicore_board_regs.v`, `link_budget_rtl.py`). `M_SCAN` (`tnca_home_pipe.v`) walks every line, 2 cycles per line, for any maintenance range:
256 lines = 512 cycles at 32 sets, 1,024 lines = 2,048 cycles at 128 sets. **How often it runs: never in a measured launch.** The board backend (`python/tnca/backends/u50.py`), the runtime and the device layer contain no maintenance command; host reads and writes go through
the coherent host port of the home slices (`hbm_read`/`hbm_write`, XRT buffer copies on the board, bench commands in the stand-in); the only users are the R40 acceptance tests (`tools/r40/cache_maintenance_tests.py`, `v3_memsys_gates.py`) and the stand-in's DMA-capability
flag (`maintenance_grep.txt` holds the commands and their output). So the longer scan prices 0 cycles on every benchmark point; it would matter only for a runtime that starts using the bulk-DMA path ("runtime bulk DMA not yet enabled",
`doc/r40_tasks_and_acceptance.md`): then 2,048 cycles per maintenance operation against 512.
Cycle effect of the larger slice: section 5. Area: section 6.

**(C) `te32`.** `TENSOR_TAG_W` (R42 lineelig) already sets the line engine's word-walk tags: `TENSOR_TAG_W 5` = 32 tags = 16 transfer slots (`NSLOT = NTAG/2`) works with today's port. `te32` = 64 tags = `TENSOR_TAG_W 6`, and `tnca_core` refuses it
(`TENSOR_TAG_W <= MTAG_W - 2`, `MTAG_W = LSU_TAG_W - 1`): the port tag is `2 + log2(LINE_SLOTS)` = 8 bits at the image's 64 line slots, so 6-bit tags need `LINE_SLOTS 128` (port tag 9 bits), i.e. twice the LSU's line-slot pool and a rotating pick over 128
slots in the LSU completion path, which is the image's worst timing family (`fast/rr_reg -> out_vector`, 24 levels, area report family 3). The lineelig engine bench (`evidence/r42/lineelig/rtl_eng/xp_tag.txt`) shows 16 against 32 word-walk tags changes only the
word engine's cycles; the line engine's are identical. Arms `te5` (`TENSOR_TAG_W 5`) and `te6` (`TENSOR_TAG_W 6` + `LINE_SLOTS 128`) are written (`tools/r42/issue/images/`); measured arm: section 5.

**(D) `hl` hardware loop / zero-cost taken branch.** Not built. ISA: no loop instruction exists (`br`, `br.nz`, `br.z`, `br.r`, `call`, `ret` only); a hardware loop would be a new encoding in `doc/isa` and a compiler change (it must be a general loop lowering).
A taken-branch predictor needs no ISA change but changes the core's front end: a taken branch is resolved in X and squashes F/D/R of its context (about 4 cycles per taken branch, probe: 6 cycles per two-instruction iteration), exactly the
path `clk` is pipelining (family 4: `branch_taken_cycles`, `park_refetch_cycles` +1). Measured ceiling (rtl-1c, 27 tiny points, counters `perf_squash`): 6,376 redirects x 4 cycles = 1.9 % of 1.34 M cycles (per point 0.2-7.9 %), so even a zero-cost taken branch with
no misprediction is bounded at 1.02x on cycles; the dse TLM key `hl`/`br0` says 1.03x on the 145 points. Ranked last.

**A second issue-side result: bank conflicts.** One read port per VRF bank (bank = register index mod 4): two vector sources in one bank take two cycles (probe table in section 1). Static count over the 25 small-tier kernel dumps of the arch worker
(`arch_dumps` in the session scratchpad; 16,849 vector instructions with two or three vector sources): 5,032 (29.9 %) read two sources from one bank, 751 (4.5 %) the same register twice (`vf.mul x, x`, squares in norms and activations: 18-33 % of such instructions in
softmax, layernorm, rmsnorm, silu, swiglu, gelu). The compiler has a bank-aware vector allocator (`bankcost.py`, `codegen.py` `bank_alloc`, knob `TNCA_BANK_ALLOC`, default 0). Measured with it on (rtl-1c, run-3 core, tiny tier, `rtl_1c/tiny_bankalloc.jsonl.gz`,
`tiny_base_vs_bankalloc.txt`): 27/27 pass, geometric mean **1.019x** (range 0.952 `softmax/fp16` to 1.055 `conv2d`; `x_vrf` 14,779 -> 11,945 on gpt2_block, 25,055 -> 10,335 on resnet_block); derived effect on the 145-point board geometric mean (`issue_predict.py`):
+1.1 % central, +1.6 % full family, +0.4 % measured points only. A hardware fix for the same-register case (one read serving both ports) was not built: 4.5 % of the multi-source instructions, in the operand-fetch cone.
Switching the knob's default is a software change outside this task's files and needs the default-change audit and `make compiler`; two tiny points get slower with it.


## 8. The combination, the staged tree and the link recipe (nothing was linked, loaded or requested)

**What the study supports as a combination.** LF (`TENSOR_LINE_HALF 1` + `MAC_F32 1`, the R42-img image `lf`) + `VRF_FWD 1` + `SETS 128` / `SET_W 7` / `SHRINK 42` (+ the area worker's `FMA_IMPL 465`, bit-identical numerics) with the software knob `TNCA_BANK_ALLOC 1` as a separate software decision.
Not in it: `TENSOR_TAG_W 5` (no gain once LF is on: section 5; +2.2 .. +4.5 k LUT), `te32` (needs 128 LSU line slots), `hl` (not built), `MTX_PIPE` (the img worker's timing signal).
Image parameter file: `evidence/r42/issue/images/issue_cand.tcl` (run-3 `image.tcl` + `TENSOR_LINE_HALF 1, MAC_F32 1, VRF_FWD 1, SETS 128, SET_W 7, SHRINK 42, FMA_IMPL 465`); stand-in form `tools/r42/issue/images/issue_cand_sim.tcl` (without `HBM_STRIPED`);
single-core device profile `core-r42-issue` (`spec/r1_profiles.json`: `core-r42-img-lf` core + `VRF_FWD 1`). Every parameter of the file reaches a parameter of `tnca_multicore_hbm_xo` (the XO packaging dry run passes it as a generic).

Gates of the candidate (functional, board-top RTL at the image's parameters): `tools/r40/image_geometry_tests.py --image-tcl tools/r42/issue/images/issue_cand_sim.tcl` (**PASS**, `gates/image_geometry_cand.log.gz`): code upload and readback, two grids of 455 programs, memory controls 1/1, 1/32, 3/1, 3/32,
coherent host window and **cache-ownership handoff** (the path that runs `M_SCAN` over 1,024 lines at 128 sets), deadline abort, final quiesce, at `SETS 128 / WAYS 8`, `SHRINK 42`, `VRF_FWD 1`, `TENSOR_LINE_HALF 1`, `MAC_F32 1`, `FMA_IMPL 465`.
Stand-in outputs of every arm are bit-identical to the run-3 image's (digest `=` in the section 5 table).

**Staged source tree** (domain: packaging, no implementation): code state commit `6d3ae693` of `R42-issue` (the RTL, profiles, targets and flow are those of `c09b6576` / `0fa0d211`; later commits change evidence/ and tools/r42/issue/ only);
`stage_6d3ae693.tar.gz` = `git archive` without `evidence/`, `doc/`, `tutorial/`, `examples/`, `deliverables/`, plus `evidence/r42/issue/images/*.tcl` and the run-3 `evidence/r41/hbm_bw/g4/run2/image.tcl`; sha256 `6fe5094c8f76f749e07626a02173cce62173dec9909a7f6cb8cc8a5882950b7f`;
on the x86 host `~/r42issue_stage/stage_6d3ae693.tar.gz`, extracted at `~/r42issue_stage/src_6d3ae693` (script `tools/r42/issue/stage.sh`). XO packaging dry run of the candidate from that tree (`flow/fpga/package_xo.tcl`, `multicore-hbm`, `TNCA_IMAGE_CONFIG` = `issue_cand.tcl`):
exit 0, `m_axi dataWidth=512 ports=3`, 104 files, 7 headers, 0 critical warnings, `tnca_u50.xo` 427,501 bytes; the log lists `VRF_FWD=1`, `SETS=128`, `SET_W=7`, `SHRINK=42`, `TENSOR_LINE_HALF=1`, `MAC_F32=1`, `FMA_IMPL=465` as image parameters (`ooc/package_cand.log.gz`).

**How the link would be started** (the recipe of `evidence/r41/hbm_bw/g4/run2/README.md` and `evidence/r42/img/staging.txt`; the author's approval, the idle rule of the card and the link budget are not part of this file):
`source ~/x86_link_env.sh; export XILINX_HLS=$HOME/Xilinx/Vitis_HLS/2023.2`; `RUN_SOURCE=~/r42issue_stage/src_6d3ae693`; `FRESH_BUILD_ROOT` must not exist;
`setsid nohup nice -n 5 bash flow/fpga/run_r40_attempt01.sh $FRESH_BUILD_ROOT $RUN_SOURCE/evidence/r42/issue/images/issue_cand.tcl`.
Clock: the image is the 100 MHz one (`TNCA_KERNEL_MHZ` default 100); at 10 ns the tile screens close with +0.58 ns (comb) and +1.01 ns (ctl) in OOC.
**Recommendation, stated plainly:** the candidate is worth a link only together with a clock above 100 MHz or with the LF image; by itself it is LF's +6-8 % on the 113 points plus the derived +1.4 % (sets128) +0.3..1.3 % (FWD) of this study, at +13.4 k LUT per tile
(SLR0 about 94 % against 89.9 % after the shrink, derived) and a tile whose worst path at 5.6 ns is 0.4 ns worse than today's (the NUCA slice). It does not meet the trigger the lead registered for image 1 (model +25 % on the 145 points).

## 9. What failed, what was not done

- **First four-context probe was wrong** (`+allroles` in `probe_run.py`): it measured the aggregate rate of four contexts as if it were one context's. Found at the first table (4 cycles per instruction), re-run with one context; the four-context numbers are kept as a result (`probe_roles{2,4}.tsv`), the first table is not.
- **v1 of `VRF_FWD` was replaced** (the forwarded value also went into the holding registers: +2.0 k LUT, +51 FF; v2: +0.9 k LUT, +1.5 k FF, same cycles on every probe). The v1 evidence is kept: lockstep `ls1/ls2` (420 runs, 0 mismatches), tile `v1_i_s3_fwd_syn` (231,441 LUT) and `v1_tscr_i_s3_fwd_p10` (+0.588 ns); the v1 screen at 5.6 ns was stopped by me after 14 minutes, by PID, when v2 replaced it.
- **Simulator builds lost under running jobs** (shared `~/.cache/tnca/devices`: a build of the same profile prunes older builds of it; and my own resync of the working tree under running jobs changed the device fingerprint): the first `VRF_PREFETCH=0` run lost its later points as
  `runtime-fault ... no simulator` (23 of 27 in the kept record `rtl_1c/tiny_nopf.jsonl.gz`, 4 pass), the first `VRF_FWD` run (3 points, identical cycles to the later run) was stopped, and every later run uses an immutable snapshot and its own `TNCA_CACHE`. The `VRF_PREFETCH 0` tier was therefore
  not completed beyond the four model points (0.904, 0.900, 0.830, 0.950 of the base) and the behavioural-VRF run (stopped by me by PID after 4 points: 1.088, 1.039, 1.045, 1.119).
- **One preserved stand-in fault**: base arm `op.fp8_linear/xl/fp8-m16-k256-n128` on thor: `DEVICE_LOST ... the completed MMIO response contains an invalid or unknown value` in `_copy_kernel` (the fault the img worker saw once on `mlp/tiny`); the record is in `rtl_3c/thor_base`, the re-run of the same point passed (61,835 cycles, the img worker's A arm: 61,751) and is the base used for that point.
- **Not built:** `hl`/zero-cost branch, `te32`, a hardware single-read for repeated source registers, W->X forwarding (one more cycle per dependent pair, a mux level in front of every execution unit), the `r_fwd` re-timing; `VRF_PREFETCH 0` and behavioural-VRF tiers beyond the points above.
- **Not measured:** medium, large, hbm and xl tiers (neither simulator reaches them); the stand-in has no point with more than 48 KiB resident outside the small tier, so `sets128` is measured where it matters least; the combination's 5.6 ns tile screen with `TENSOR_TAG_W 5` (`comb5`) was still running when this file was last edited, as was the 6.67 ns screen of fwd2 (the 6.67 ns screen of comb was not run).
- **Model-side:** I did not run new TLM what-ifs (the TLM's `issue_vector_cycles` has no counterpart mechanism in the RTL, so a "true issue cost" run would be another calibration); the model comparison is per point against RTL counters and measured effects.
- `make r1-profile-tests` fails CFG-01 in this tree for the benches of the other R42 profiles (`tb_core_r42*.vvp` not built here), as before this branch; the lines of `core-r42-fwd` read back PASS.
- Resource use: GB10 and thor up to 4 and 3 jobs each (briefly 5 on GB10 for a few minutes when two arms overlapped), x86 at most 2 own Vivado jobs (plus a 3 minute packaging dry run overlapping the tail of a synthesis); processes of mine that were stopped, by explicit PID: `~/r42/issue_out/killed_own_pids.txt` on GB10/thor, `~/r42_issue_out/killed_own_pids.txt` on x86. No pattern kill, no foreign process touched.

## 10. Remaining ideas, ranked by expected gain (derived/measured sizes as above; none is built)

1. **Bank conflicts: switch `TNCA_BANK_ALLOC` on by default after the gates** (+1.1 % central derived, +1.9 % on the tiny tier, no hardware, no timing risk). Then look at why `softmax/fp16` gets 5 % slower with it and fix the allocator's heuristic (the loss is in the allocator, not in the hardware). Next: a compile-time rule that a vector instruction's two sources are different registers when a copy is cheaper than a stall (the 4.5 % of instructions that read one register twice).
2. **Make `sets128` pay for itself in timing**: the cycle gain is real (stand-in small tier -7 .. -33 %, derived +1.8 % central on the board mean, model +4.7 %); the cost is +11.5 k LUT per tile and 0.4 ns at 5.6 ns in the NUCA slice's victim/valid/write-back cone. The slice's cone needs the `clk` worker's registered tag read and a registered
   victim select first; `SETS 64` (32 KiB per slice) is the untested intermediate (the stand-in can price it: `SETS 64 SET_W 6` is an image parameter, `cache_sets` accepts it).
3. **`VRF_FWD`**: +1.3 % central derived (rtl-1c), +0.4 % on the stand-in, +0.9 k LUT, `r_fwd` path -0.69 ns at 5.6 ns (not the limiter). Worth keeping in the RTL and in the profile; worth a link only inside a bundle. To lower its timing cost: take `stall_front` out of `r_fwd`'s input (`r_fwd = r_whit & w_full` at the point of use).
4. **W -> X forwarding** (one more cycle per dependent pair: pure RAW 2 -> 1 cycle, the behavioural bound is 1.04-1.12x on the model points against 1.02-1.07x for VRF_FWD): one mux level in front of every execution unit's operand, i.e. in the FMA stage 0 and the VALU: not for a 5.6 ns tile.
5. **Taken-branch refill** (ceiling +1.9 % of cycles, 4 cycles per taken branch) and 2-wide issue: after `clk`'s front-end re-pipelining, not before.
6. `te16`/`te32`: no gain once LF is on (section 5); only for an image without LF.
