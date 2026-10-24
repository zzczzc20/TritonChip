# R41 Execution Checklist: Frequency and Dual Memory Paths

Date: 2026-10-20. Author instructions: branch `r41` first, list tasks, then implement.
This checklist records the execution scope for this iteration; the existing [r41_tasks_and_acceptance.md](r41_tasks_and_acceptance.md) is a broader unfrozen draft.

## Starting Point and Status

- [x] Create `r41` from R40 closeout commit `9235f5d`.
- [x] Merge `r40-final-image` commit `6eed3e0` after listing tasks for the two main tracks.
- [x] Complete M1–M4 related regressions on merged R41 source code.
- R40 retains status `closed-with-failures`, `R40-10.acceptance` not passed.
- Baseline loaded image remains `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, measured at 90 MHz, 3 cores, 3 HBM outlets, 4 GiB window. (Board observation noted an unverified 245 MHz image; B40-OBS cannot be assumed active without an authorized reload).
- Uncommitted SystemC/Ramulator2 implementation and R41 contract drafts in the original workspace are preserved as-is, not automatically treated as verified deliverables.

## Track 1: 95–100 MHz

M1–M4 have been merged into source code, but have not yet been implemented in a new image or brought up on-board:

| ID | Description | R41 Verification |
|---|---|---|
| M1 | Host grant registered boundary | Host mux directed, random, and 3-core ownership tests |
| M2 | Source and tile-local `host_allow` registers | Coherent-host, permission revocation, and in-flight responses |
| M3 | Local memory controls, aperture precomputation, MMIO settle | Memory mode, MMIO, AXI limits/stages, actual image geometry |
| M4 | ISA §8.4 float-to-int fix | Independent Fraction oracle, edge vectors, and core instruction regressions |

Task sequence:

1. Complete the above regressions and archive current source identity and logs (completed in `merged-boundaries-summary.json`).
2. Reconstruct path classification from the raw census, verifying complete coverage across 12,799 endpoints / 753 families (completed in [Stage 1 frequency preflight](../evidence/r41/stage1/frequency_preflight.md)).
3. Determine register boundaries category-by-category according to the table below, screening with unit function and iso-constraint out-of-context (OOC) synthesis first.
4. After integrating candidates, rerun associated core/protocol regressions, resource and placement preflights, and update the census.
5. Decide the full implementation based on verified candidates; measure frequency and correctness on board only after completing route timing.

### Approximately Ten Structural Candidate Changes

This is a work package to be designed and screened; the ten entries must not be equated to proven timing closure.

| ID | Residual Structure | Modification and Verification Focus |
|---|---|---|
| F01 | `stall_front` / operand enable high fanout | Local controls and credited boundaries; preserve in-flight instruction, flush, and replay semantics |
| F02 | Scalar read → barrier / scoreboard / issue | Split read values and dependency checks into stages; verify cross-role wakeup and RAW |
| F03 | VFPU `red_busy` → FMA stage-0 | Register arbitration / input boundaries; verify alternating reduction and standard FMA |
| F04 | Writeback `w_rctx` / `w_en` → `w_sdata` | Cohort context with data; verify multi-context conflicts and return ordering |
| F05 | Scoreboard `sb_v` → VRF bank / collector | Local valid bits and collection boundaries; verify bank conflicts and forwarding |
| F06 | Dispatch DSP `total` → `remaining` | Decouple multiplication results from dispatch state updates; verify dimension boundaries and deadlines |
| F07 | Slice `cur_ms` / `ms_way` → data RAM | Audit existing C1 candidate `658c811`; preserve commit, eviction, and maintenance visibility |
| F08 | XLS FMA `p0_sticky_al` → `p1_sh_mag` | Rebalance generation source and pipeline; preserve IEEE rounding and per-K accumulation ordering |
| F09 | XLS FMA `p1_sh_mag` → MTX accumulator | Accumulator writeback boundary and associated latency; verify continuous MMA, ACC RAW, source-free/result-ready |
| F10 | LSU `pending` / FSM → mesh input FIFO | Request/credit register boundaries; verify backpressure, fault drain, and response ownership |

References: [Raw census and analysis notes](../evidence/r40/v3/final_image_candidate/census/README.md), [Frequency frontier](../evidence/r40/v3/final_image_candidate/census/frontier.json), [Stage 1 frequency preflight](../evidence/r41/stage1/frequency_preflight.md).
The 12,799 count is the what-if analytical result on the raw routed DCP after excluding M1–M3 combinational cones, not the measured failure count of the merged new design. 95 MHz corresponds to 170 blocking families and 8,959 endpoints; 100 MHz corresponds to all 753 families. Temporary path exclusions from this analysis must not be carried into implementation constraints.

## Track 2: Dual Memory Paths

- [x] Audit and build existing implementation; pin Ramulator2 revision (`c6c7a5de0cef35935d6fad7369e6271406798865`), configuration, and source identity.
- [x] Compare default fast path against R40 M2 for architectural results, cycles, and counters under monitor mode invariance.
- [x] Verify slow path for row hit/miss/conflict, bank parallelism, read/write turnaround, refresh, queue backpressure, and write completion across nine bridge mechanisms.
- [ ] Check authentic address → PC/bank/row mapping and AXI/NoC/home/controller/DRAM timing boundaries, avoiding double counting (address-to-PC translation and maintenance drain fix verified; physical bank/row mapping and controller timing reconciliation remain open).
- [x] Verify path switching (`TNCA_DRAM_PATH=fast|slow`) and explicit refusal when Ramulator2 dependency is missing or requests are out-of-domain.
- [x] Verify replay / fixed-point iteration convergence, request matching, and non-converged states; audit scope of per-command DRAM state reset (child process lifetime and death cleanup verified via `PR_SET_PDEATHSIG`; non-converged GEMM cases preserved).
- [ ] Preserve separate development/calibration/validation/holdout data; compare residuals and simulation wall time (sealed holdout untouched; matched execution speed measured; full-chip timing calibration remains uncalibrated).
- [ ] Update model configurations according to new RTL pipeline; record model and hardware identity with each result (tracks Stage 2 new RTL image delivery).

Implementation entry points: `tools/performance_harness/systemc/include/tnca_dram.hpp`, `src/chip.cpp`, `ramulator/` (relative to `tools/performance_harness/systemc/`).
The current slow path is a per-command fork/replay fixed-point model; online DRAM coupling or full timing calibration must not be claimed before formal verification is complete.

References: [Dual DRAM paths validation](../evidence/r41/stage1/memory_paths.md), [BO/physical-PC address translation](../evidence/r41/stage1/address_map.md).

## Completion Evidence

Retain source identity, configuration, command, exit code, and results for each item.
Functional pass, out-of-context (OOC) timing, full-chip route timing, on-board frequency, and model calibration are judged separately.
Historical R40 PASS records serve as provenance evidence and must not be directly copied as R41 PASS records.
