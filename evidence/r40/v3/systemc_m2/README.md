# SystemC/TLM model, milestone M2: timing fidelity and calibration against the board

**Verdict (doc/r40_tasks_and_acceptance.md sec 3.4): UNCALIBRATED.** On the frozen holdout split (39 configurations of 19 points) the median absolute graph-makespan error is 21.5 % and the p95 34.7 % (limits 10 % / 20 %; mean signed error -16.5%). The model's cycles therefore never replace a board measurement. On the calibration split it was fitted on: median 9.1 %, p95 26.3 % (60 configurations).

Judged with `tools/performance_harness/systemc/calibrate.py evaluate` (unchanged) against the board timing dataset; the fit report declares `fit_rows: calibration-only` and no holdout or unlabelled row was read by any fitting step. Nothing was refitted after the holdout was evaluated.

## 1. Core timing fidelity (RTL as the authority for structure)

All of it sits behind `core_pipeline=1` / `memory_model=1` in `configs/r40-board-m2.cfg`; `configs/r40-three-core.cfg` keeps the M1 behaviour bit for bit (every M1 validation passes unchanged, section 5). Every constant is a configuration key; `configs/r40-board-m2.sources.json` gives the RTL source of each of the 94 keys.

| gap (M1 list) | RTL structure modelled | where |
|---|---|---|
| store blocking / LSU | two load/store units (`LSU_INFLIGHT=2`, tnca_lsu_pair.v): the second takes an access only after the first sent all its beats; first beat at c+2, one beat a cycle on the core memory port shared with the transfer engine; a unit is busy until its last response (stores wait for the write response) + 3; load-use = last response + 5 (+1 vector); a busy unit holds the access in R and freezes every context; LSU completions take the single writeback port (wb_steal) | `Core::execute` K::Lsu, `units`, `steals` |
| register-bank contention | banked VRF, 4 banks, one read per bank per cycle (tnca_operand_collector.v); sources per tnca_decoder.v (`o_va/vb/vc_en`, a masked write does not read its destination); `VRF_PREFETCH=1`: the read in D hides one conflict when R is not reading | `Core::bank_conflicts` |
| instruction fetch | a taken branch costs its context 4 fetch slots; `RCTX_SWITCH=2`: a register hazard (or a busy matrix/transfer engine) parks the context, which refetches 5 cycles after its wake | `Frame::fetch_ready`, `Core::hazard` |
| round-robin issue arbitration | one in-order pipeline per core shared by the four role contexts, one instruction a cycle in request order; stalls that do not switch (LSU busy, SFU, fence, arrive, bank conflict) freeze the pipeline for every context (`stall_front`); fence/arrive drain the whole LSU (`lsu_req_ready`) | `Core::issue_slot`, `freeze`, `hold` |
| DMA beat width | 8-byte units (MEM_DW 64); setup 2 + rank; loads hold one of 8 slots (NTAG/2) until written to the window; stores read each 64-byte window register (S_SREAD, 3 cycles) then one word a cycle with 16 tags; drain 2; transposed loads per S_XRUN/S_XDRAIN/S_XWRITE: one column block at a time, one unit per row, drain, four row writes, no overlap; one command in the engine (`stall_txf`) | `TensorEngine::fast_transfer` |
| memory system | home slice engine per tnca_nuca_slice.v states (hit 3, miss 4+2+2+2/target, dirty victim +2), MSHR 8, round-robin victim skipping pending fills; one-flit packets, 3 cycles/hop; outlet issue 2 (read) / 3 (write) cycles, `TNCA_RTL_OUTSTANDING` lines held until the last beat / write response; `TNCA_RTL_MEM_PORTS` | `include/tnca_timeline_memory.hpp` |
| matrix busy cycles | from the M1 RTL oracle: 2a+18K+6 with ABUF (FP16 K16: 310), 34K+6 unbuffered, m.zero 17, a.vwrite 2, row ops 7 | config `mtx_*` |
| host time | in server mode a fixed host gap before every launch and a per-KiB staging cost per host transfer (fitted, section 2) | `Chip::run_command`, `host_charge` |

Other RTL latencies now taken from the RTL: vf.cmp -> predicate 4, extract 1, integer reduction 1 (scalar) / 3 (vector), FP reduction 16 / 17, SFU result 28 with the next pass at 27, 90 MHz qualified clock.

**A real bug found and fixed.** At 90 MHz the cycle period (11111.1 ps) is not a whole number of 1 ps units, so each wait rounded differently and two events of the same cycle could compare unequal: a transfer engine register write due at cycle N was not yet committed when an MMA read the register at cycle N (llama_prefill/tiny gave NaN outputs with the timeline memory). Fixed in include/tnca_model.hpp Context: the period is rounded once to whole resolution units (11111 ps); every model time is an integral number of cycles. Configs at 100 MHz (exact) are unaffected.

**Simulation speed** (needed to run 284.9 G board cycles of calibration and holdout configurations): a conservative lookahead (`time_quantum`): a context skips a wait that ends before any other pending activity of the simulation, which changes nothing but the order of same-cycle events; the cold call (compile, setup) of a multi-launch graph runs untimed (`T 0`, architectural effects only, home-slice contents kept) and timing starts at the measured steady call; a one-launch graph is measured on the cold call's graph launch (no steady call); `require` no longer builds a string per check. Deviations measured with the final binary are in section 5. Final campaign: 195 runs, 256 G modelled cycles in 9.6 process-hours; per run 5.9 M cycles per second of model time (median; 1.3-51.7 M), 11.5 M median on the 23 runs above 1 G cycles. M1 ran 1-8 M cycles/s per run.

## 2. Fit (calibration split only)

Procedure (`fit.py`, `fit_report.json`):

- structural constants set from the RTL of the attempt03 board image (no fitting)
- matrix busy cycles set from the RTL oracle (2a+18K+6 with ABUF; m.zero 17; row ops 7)
- host: median calibration inter-launch gap -> host_launch_cycles; median calibration per-kernel staging rate -> host_transfer_cycles_per_kib
- latency: calibration configurations simulated at the grid [[20, 5], [20, 40], [35, 5], [35, 40], [55, 5]] of hbm_latency_cycles, tl_outlet_to_slice_cycles; per-configuration (bi)linear interpolation; objective: max(median/0.10, p95/0.20) of calibration |graph makespan error|, then median
- tl_hbm_write_latency_cycles tied to hbm_latency_cycles
- holdout evaluated once with calibrate.py at the frozen configuration

| fitted key | value | from |
|---|---|---|
| host_launch_cycles | 405467 | median of 18 inter-launch gaps of the calibration records' measured calls (p05 365754, p95 968706; 4.5 ms at 90 MHz: host submission, including the runtime queue's 2 ms poll) |
| host_transfer_cycles_per_kib | 4448 | median of 35 calibration per-kernel records' steady upload+download seconds per byte (p05 4362, p95 4569) |
| host_transfer_cycles | 0 | a fixed cost per transfer is not identifiable from the calibration records (all MiB-sized); see directional results |
| hbm_latency_cycles = tl_hbm_write_latency_cycles | 23 | AXI/HBM round trip behind an outlet |
| tl_outlet_to_slice_cycles | 13 | memory return path outside the outlet's outstanding window (RTL-derived 5) |

Latency grid: every calibration run (49 distinct point/knob configurations; a configuration recorded under two software trees is run in each) except two 1-core GEMM_TILE=16,32,16 configurations of 53.5 G and 43.7 G board cycles, left out of the grid for cost and included in the final evaluation:

| hbm_latency_cycles : tl_outlet_to_slice_cycles | median abs | p95 abs | max(median/0.10, p95/0.20) |
|---|---|---|---|
| 20:5 | 10.8 % | 28.2 % | 1.41 |
| 20:40 | 6.4 % | 26.1 % | 1.30 |
| 35:5 | 9.6 % | 37.1 % | 1.85 |
| 35:40 | 11.2 % | 61.9 % | 3.10 |
| 55:5 | 9.6 % | 97.9 % | 4.90 |
| **chosen 23 : 13** (interpolated) | 8.6 % | 23.2 % | 1.16 |

The objective is the sec 3.4 test itself (both limits), not the median alone: the median-only optimum (22 : 36) trades a lower median for a much larger tail. Even at the chosen point the calibration p95 exceeds 20 %: no single latency fits both the latency-bound transfer-engine families (matmul, linear, gemv, which want a short round trip) and the load/store streaming families (silu, rmsnorm, swiglu, transpose, which want a long one). The confirmation run at the chosen point gives the calibration numbers above.

## 3. Holdout and where the model is wrong

| split | family | configurations | median abs | max abs | mean signed |
|---|---|---|---|---|---|
| calibration | attention_decode | 1 | 9.2 % | 9.2 % | -9.2% |
| calibration | bmm | 1 | 7.4 % | 7.4 % | -7.4% |
| calibration | conv2d | 1 | 8.7 % | 8.7 % | -8.7% |
| calibration | gelu_tanh | 1 | 4.2 % | 4.2 % | -4.2% |
| calibration | gemv | 5 | 6.1 % | 102.3 % | +17.4% |
| calibration | gpt2_block | 1 | 1.2 % | 1.2 % | -1.2% |
| calibration | linear | 8 | 9.7 % | 31.5 % | +3.9% |
| calibration | matmul | 6 | 6.0 % | 12.5 % | -4.0% |
| calibration | rmsnorm | 9 | 14.3 % | 17.7 % | -14.5% |
| calibration | silu | 9 | 12.4 % | 26.3 % | -13.7% |
| calibration | softmax | 10 | 3.9 % | 8.1 % | -1.3% |
| calibration | swiglu | 1 | 16.2 % | 16.2 % | -16.2% |
| calibration | transpose | 7 | 10.8 % | 21.0 % | -8.1% |
| holdout | add | 10 | 26.9 % | 30.5 % | -26.0% |
| holdout | argmax | 1 | 18.8 % | 18.8 % | -18.8% |
| holdout | attention_prefill | 1 | 11.1 % | 11.1 % | -11.1% |
| holdout | copy | 9 | 28.6 % | 36.4 % | -29.0% |
| holdout | embedding | 2 | 6.1 % | 6.1 % | -6.1% |
| holdout | gemv | 3 | 6.1 % | 6.1 % | -3.6% |
| holdout | layernorm | 1 | 13.6 % | 13.6 % | -13.6% |
| holdout | llama_decode | 2 | 0.8 % | 1.0 % | -0.1% |
| holdout | llama_prefill | 2 | 3.6 % | 4.2 % | -0.7% |
| holdout | matmul | 3 | 5.8 % | 5.8 % | -4.4% |
| holdout | mlp | 1 | 14.8 % | 14.8 % | +14.8% |
| holdout | resnet_block | 1 | 8.7 % | 8.7 % | -8.7% |
| holdout | rmsnorm | 1 | 10.9 % | 10.9 % | -10.9% |
| holdout | rope | 1 | 24.3 % | 24.3 % | -24.3% |
| holdout | silu | 1 | 12.3 % | 12.3 % | -12.3% |

**Load/store streaming is too fast (the holdout failure).** Every family that moves its data with vector loads and stores is under-predicted: add and copy by 21-36 % (they are pure streams and were not in the calibration split), rope, argmax, layernorm, silu, rmsnorm, swiglu and transpose by 10-25 %. The holdout is dominated by add and copy (19 of its 39 configurations), hence its median. The same memory parameters predict the transfer-engine families (matmul, gemv, softmax, embedding, both llama graphs, gpt2) mostly within 6 % (matmul up to 12.5 %), so the missing time is on the core-side load/store path, not in HBM: on the board one 32-byte vector access of copy/hbm on 3 cores occupies an LSU unit about 120 cycles, in the model about 90. The latency fit cannot absorb it: the second latency parameter (the memory return path) would slow the streams but equally the transfer engine, and the sec 3.4 objective keeps it small. Which RTL stage the model is missing (LSU beat issue, write-allocate store path, core memory port, response writeback) is not identified; it needs an RTL trace of a streaming kernel, which this milestone did not take.

**Linear with transposed weights depends on shape.** linear/hbm on 3 cores is 31 % too fast while linear/large on 3 cores is 18 % too slow and model.mlp/large (two such GEMMs) 15 % too slow; the 1-core 1-outstanding linear configurations are 9-10 % too slow. The weights are read with the corner-turning transposed transfer; rows of the B tile are one row pitch apart, so they fall in one cache set and the slices thrash. How much they thrash depends on the pitch and the round-robin victim order, and the model reproduces it only approximately. The extreme case is GEMM_TILE=16,32,16 on 1 core: gemv is predicted 102 % too slow (the board pays 3.3x over the 16,16,16 tile, the model 7x), while linear and matmul at that tile are within 3 % and 13 %.

**What is right.** Matrix-engine and transfer-engine bound graphs (matmul, gemv, bmm, conv2d, attention, softmax, embedding) are within about 9 % (apart from the 16,32,16 gemv above), and the multi-launch graphs that combine them with the fitted host gap (llama_decode/hbm and /large, llama_prefill, gpt2_block) within 1-4 %.

## 4. Directional validation (A/B effects measured on the board)

Pairs of board configurations of one point and software tree differing in exactly one knob; board effect = ratio of graph makespans, model effect = ratio of the two predictions; a direction counts as reproduced when both ratios fall on the same side of 1 or both within +-2 %.

| knob (a -> b) | pairs | direction reproduced | median board ratio b/a | median model ratio b/a | median abs ratio error |
|---|---|---|---|---|---|
| active cores 1 -> 3 | 6 | 6 | 0.362 | 0.377 | 5.6 % |
| memory ports 1 -> 3 | 6 | 3 | 1.002 | 0.967 | 4.4 % |
| outstanding 1 -> 8 | 6 | 6 | 0.948 | 0.897 | 7.1 % |
| per-kernel -> resident | 48 | 48 | 0.349 | 0.567 | 45.9 % |

Active cores (1 -> 3) is reproduced in direction and size on all six pairs. Outstanding (1 -> 8) is reproduced in direction on all six, but the model gains more from eight lines than the board on the streams (copy 0.84 vs 0.94): the board stream is limited elsewhere, consistent with the streaming deficit above. Memory ports (1 -> 3): the board shows no effect on add, copy, silu and rmsnorm (within 1 %) and a large one on transpose (0.52); the model predicts the transpose gain (0.41, larger) but also a 3-8 % gain on add, copy and silu that the board does not show, so three of six directions are counted as not reproduced.

Resident vs per-kernel transfers (48 tiny/small/medium multi-launch pairs, the "other" split) is reproduced in direction on every pair, but the model under-states it (board median ratio 0.35, model 0.57): the board pays a fixed cost per host transfer that dominates small tensors, and host_transfer_cycles is 0 because the calibration records (all MiB-sized transfers) cannot identify it.

## 5. Validation of the changed model

- Default configuration (M1 behaviour): validate_numeric rc=0, validate_chip rc=0, validate_roles rc=0, validate_dma rc=0, validate_matrix rc=0, validate rc=0, validate_monitor rc=0.
- Transfer engine against the independent rank/stride/byte oracle (validate_dma.py, transposed included) on the timeline memory with core_pipeline=1: rc=0.
- End to end through torch.compile -> Inductor -> Triton -> artifact -> Session with the board configuration: tiny suite 27/27 pass (3 active cores), tiny+small 52/52 pass (3 ports, 8 outstanding), outputs compared with PyTorch at each point's tolerance. Every final calibration/holdout run also passes its numeric check.
- Lookahead vs every wait taken, 52 tiny/small graphs: 15 identical, median |difference| 0.1 %, max 4.1 % (op.gemv/small/fp16-k256-n512): only the order of same-cycle events differs; contended small kernels amplify it, the large calibration points moved by at most 0.04 % under lookahead.
- Untimed vs timed cold call (steady-call cycles), 52 tiny/small graphs: 28 identical, median |difference| 0.0 %, max 5.9 % (op.gemv/small/fp16-k256-n512): the timed call starts from the slice contents an untimed cold call leaves (lines placed, no timing state), and same-cycle order differs; contended small kernels amplify it, the large calibration points moved by at most 0.04 % under lookahead.
- Timeline vs the M1 detailed memory path (core_pipeline=0): median |difference| 3.2 % over 52 tiny/small graphs (1 port, 1 outstanding), max 142.9 % on op.transpose/small/fp32-r128-c256. memory_model 0 (M1 detailed TLM path, validated against RTL captures for isolated accesses) vs 1 (timeline), core_pipeline=0, time_quantum=0. The timeline charges the home-slice engine the RTL engine states (3 cycles a hit, 4+2+2+2 a miss) where the detailed path charges service_cycles=1; under load the slice engine becomes the bottleneck on conflict-heavy patterns (transpose).

## 6. Limits of this result

- Uncalibrated: by sec 3.4 the model's cycles are a diagnosis tool, not a measurement substitute. The per-family table says where its numbers can be read (engine-bound kernels and the llama graphs) and where not (load/store streams, transposed-weight linear at the tile shapes above).
- Two calibration configurations (1 core, GEMM_TILE=16,32,16, 53.5 G and 43.7 G board cycles) were left out of the fitting grid for cost; they are in the final calibration numbers (linear -2.9 %, matmul -12.5 %).
- An exploratory first latency grid run before the transposed-transfer sequencing was made RTL-exact was discarded; only the grid listed above, made with the final binary, entered the fit.
- Lookahead and the untimed cold call are not bit-exact against every-wait timing (section 5); one-launch graphs are measured on the cold call's graph launch, multi-launch graphs on a steady call as on the board.
- The dataset adds the fe1b campaign records copied read-only from the FPGA host up to 11:20 UTC (615 passing 90 MHz records: 61 calibration, 40 holdout, 514 other); the coordinator's 373-record dataset is a subset. Several board configurations share a (point, knobs) across software trees; calibrate.py pools their board records, so such a configuration is counted once per tree.

## Files

| file | content |
|---|---|
| `README.md` | this summary |
| `fidelity_report.json` | calibrate.py evaluate: every (prediction, board) pair, split statistics, the verdict |
| `fit_report.json` | procedure, fitted values and the source of all 94 configuration keys; fit_rows = calibration-only |
| `host_fit.json` | host gap and staging fit from calibration records (inputs listed with sha256) |
| `latency_fit.json` | latency grid, per-configuration model and board cycles at every grid point, interpolated scan |
| `predictions_final.jsonl` | calibration + holdout predictions at the frozen configuration |
| `predictions_other.jsonl` | per-kernel/resident pairs of the other split |
| `grid/pred_*.jsonl` | calibration predictions at each fitting-grid point |
| `directional.json` | every A/B pair (board and model ratios) and the per-family breakdown |
| `validation_m2.json` | validations of the changed model and exactness checks |
| `speed.json` | wall time and modelled cycles per final run |
| `runs.json` | status of every final and grid run |
| `identity.json` | binary, configuration, dataset and harness source sha256 |
| `board_timing_dataset_m2.json.gz` | the board dataset used (board_dataset.py over the bc5a/80dd/fe1b records) |
| `r40-board-m2.cfg` | the frozen board configuration (copy of tools/performance_harness/systemc/configs/r40-board-m2.cfg) |
| `raw_tarball.json` | location and sha256 of the raw runs on GB10 |

Raw runs (archbench records, model counters/launches/identity, run_graphs results, child logs, kernel assembly, configs) of every final and fitting-grid run: GB10 `/home/zhongchun/m2/m2_raw_evidence.tar.gz` (6727636 bytes, sha256 `c9b396cf6f8c6f92855e051f74932651ee1766e3f1f647c6b4b4b140fb6da181`).

Model binary sha256 `2402d46d9375558c5fc80d2689b9b03fc235817902d2bf38852492f4d6a3c98d`, configuration `configs/r40-board-m2.cfg` sha256 `dc2c1763eafb7d022304ee820651465c4df04135a68381c0d3833ea7195eb796`; the harness sources are listed with their sha256 in `identity.json`. No RTL simulator or Verilator is in the model's execution path; GB10 only (at most 8 cores); the FPGA host was only read (campaign JSONL copied); nothing committed.
