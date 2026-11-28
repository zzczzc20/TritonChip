# R42 issue: results (worker `issue`, branch R42-issue, base R42 129f213d)

Domains in every number: **rtl-probe** (core testbench, assembly microbenchmarks, one core), **rtl-1c** (single-core RTL simulation of ArchBench points,
`archbench.run --counters`, profile `core-vrf-sync-v5-x8-g2c4-dp`), **rtl-3c** (three-core board-top RTL stand-in at the run-3 image geometry, `tools/r42/img_mc.py`),
**OOC** (Vivado 2023.2 out-of-context synthesis / place and route on xcu280, x86 host), **TLM** (the dse worker's SystemC model; its files were only read),
**derived** (arithmetic on recorded numbers). No board number exists for anything here. No link, card load or board access was made.

## 0. Findings in one paragraph each

1. **The 2-cycle-per-vector-instruction mechanism of `vi1` does not exist in the run-3 core.** With `VRF_PREFETCH 1` + `VRF_BYPASS 1` (the image's parameters) independent vector
   instructions of 20 kinds (0-3 VRF sources, int/FP/convert/shuffle) issue at exactly 1.00 cycle each; 2.00 appears only with `VRF_PREFETCH 0`. What costs cycles is
   (a) a dependent consumer: producer in W, consumer reads the bank in R (request cycle + response cycle), (b) two sources in one VRF bank (+1 cycle per extra source),
   (c) refill after a taken branch. The TLM's `issue_vector_cycles 2` is a calibration knob, not a mechanism: its tiny-tier `vi1` effect is 1.165x, the RTL bound for removing every operand-fetch
   stall cycle is 1.067x (geometric mean over the same 27 points).
2. **`VRF_FWD` (built, default off) is the cheapest change that overlaps the operand fetch of a dependent instruction**: the writeback register is forwarded into R's operand capture.
   Measured (rtl-1c tiny, 27 points): 1.021x geometric mean (1.025x cycle-weighted), derived +1.3 % on the 145-point board geometric mean (central). Price: +0.9 k LUT per tile (OOC synthesis), timing at 5.6 ns see section 6.
3. **The larger issue-side loss is bank conflicts, and the compiler already has the fix, off by default (`TNCA_BANK_ALLOC=1`)**: 1.019x on the 27 tiny points (rtl-1c), derived +1.1 % central, no hardware.
4. **`hl` / zero-cost branch: not built.** Upper bound 1.9 % of cycles (6,376 redirects x 4 cycles over 1.34 M cycles of the 27 tiny points); it needs a front-end change in the path that `clk` is re-pipelining; no ISA change is needed for a predictor but a hardware loop is a new encoding.
5. **`te32` does not fit this core as modelled**: the line engine's word-walk tag width is bounded by the port tag (`TENSOR_TAG_W <= port tag - 3`), the port tag is 2+log2(LINE_SLOTS) = 8 bits, so `te32` (6-bit tags, 32 slots) needs 128 LSU line slots, i.e. a wider LSU completion pick (the image's worst timing family); `te16` (`TENSOR_TAG_W 5`) exists and needs nothing.
6. **`sets128`**: image parameters only (`SETS 128`, `SET_W 7`, `SHRINK`). No benchmark point issues a cache-maintenance operation, so the 2,048-cycle `M_SCAN` costs 0 cycles in every measured launch.

(Sections 1-9 carry the numbers; section 10 is the list of what was not done.)

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

## 2. What was built and changed (all default-off; every default keeps the shipped RTL bit- and cycle-identical, see section 4)

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
