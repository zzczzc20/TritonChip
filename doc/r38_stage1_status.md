# R38 Stage 1 status

Plan and acceptance: [r38_stage1_tasks_and_acceptance.md](r38_stage1_tasks_and_acceptance.md).
Updated 2026-10-08. Stage state: **IN PROGRESS / NOT ACCEPTED** (gates G1a, G1b open on the items marked below).

Generated evidence lives under `sim/build/r38/` and is not in Git.

| WP | State | Evidence / command | Open |
|---|---|---|---|
| S1-00 evidence registry | **PASS** | `python3 tools/r38/evidence_tests.py` (15 entries, 6/6 mutations caught) | — |
| S1-01 baseline | **PASS** | `make r38-baseline`: 447/447 (O0/O1/O2 × 149). `make r38-baseline-check`: two O2 runs identical to `spec/r38/baseline_cycles.json` through the official device path, after the RTL changes below | — |
| S1-02 profiler v0 | **PASS** | `make r38-profiler`: identities on every program of all 149 O2 cases, `core-sim` and `core-vrf-sync-v1`; cycles equal to the frozen baseline; old device reports unavailable; 2/2 mutations caught. `r36-rtl-tests` PASS; `r36-build-tests` PASS on the x86_64 acceptance host (`sim/build/r38/r36_build_linux.log`) | — |
| S1-03 records/manifest | **PASS** | `tools/r38/manifest_tests.py` | — |
| S1-04 smoke/dev | **PASS** | smoke: 9/9 cases × 3 repeats identical, 836 s of a 900 s budget. dev: 157/157 points, 13 operators, 46 families, 51 min | a faster simulator (notes) |
| S1-05 ArchBench v0 | 17 pass, 1 pending | `python3 tools/r38/archbench_tests.py`; η ≤ 1 on all bounded baseline, smoke and dev points | §4.10.9-7 needs degraded RTL builds |
| S1-06 cost model v0 | **ACCEPTED** (with a scope limit) | `tools/r38/costmodel/evaluate.py --final` run once: held-out n=26, median 0.0 %, P90 7.3 %, Spearman 0.999, coverage 0.88 | held-out split contains no GEMM/transpose/softmax/rmsnorm family |
| S1-07 three-layer feature | **PASS** | `tools/r38/loop_tests.py` 6/6; `make r38-xpose-rtl`; R37 447/447 on `core-sim` (cycles identical at O0/O1/O2) and 447/447 on `core-xpose-v1`; record `spec/r38/design_changes/xpose-dma-001.json` | — |
| S1-08 case ablations | — | — | after S1-07 |
| S1-09 gates | L0/L1/L2 + selftest implemented; selftest **PASS** | `tools/r38/gate.py l0\|l1\|l2\|selftest`; log `sim/build/r38/devops_log.jsonl` | an L2 promotion run |
| S1-10 PyTorch (P1+, replaces the P0/P1 plan) | **IN PROGRESS**: ArchBench(torch) tiny + small 52/52 compile with zero host fallback and launch counts equal to the generated wrapper's; 7 regression graphs added for the fixes (incl. a device-assert trap) PASS; single-query (decode) attention fixed (three boolean defects, §6 of the P1+ doc); **tiny tier on the RTL 27/27 pass** (section below); 24/24 graph tests + the assert trap | [r38_pytorch_p1plus.md](r38_pytorch_p1plus.md) §6; `make r38-torch`, `make archbench-coverage`, `make archbench-tiny` | Agent pilot not started |
| S1-10 NVIDIA scoring rules | **written**: [archbench/PROTOCOL.md](../archbench/PROTOCOL.md); exercised on GB10: 76/76 points under bound v2, η ≤ 1 throughout (bound v1 had found an η = 1.2 audit failure) | `make archbench-gpu`, `python -m archbench.report` | GB10 tensor-core peak is an upper estimate |
| Board (Stage 2 hand-off) | **U280 loaded, ladder B1–B6 passed** (2026-10-08) | [r38_board_bringup.md](r38_board_bringup.md) §5: conformance 5/5 measured, profiler (full), ArchBench tiny 27/27 + small 25/25, tiny cycles identical to the RTL launch by launch, multi-step LLaMA generation | board evidence is `measured-board` in ArchBench records; the evidence registry entry stays `documented` until Stage 2 levels apply |
| S1-11 related work | — | — | — |

## 2026-10-07: ArchBench (PyTorch) tiny tier, TNCA vs GB10

Same 27 points, same seed, each platform's own `torch.compile`; balanced weights; strict
bounds `archbench-torch-bounds/2`. TNCA: `core-sim`, O2, device cycles summed over launches
(serial-grid), launch counts equal to the generated wrapper's. GB10: CUDA-graph replay,
CUPTI device time, cold caches. Both full suites certified.

| | pass | E (harmonic η) | A (arithmetic η) |
|---|---:|---:|---:|
| GB10, `inductor-cudagraph` | 27/27 | 0.0028 | 0.0068 |
| TNCA `core-sim`, RTL | 27/27 | 0.0091 | 0.0293 |

Models on TNCA (RTL cycles, η): LLaMA prefill 3.88 M (0.0099), LLaMA decode 2.57 M (0.0091),
GPT-2 block 2.16 M (0.0117), MLP 1.14 M (0.0145), ResNet block 0.66 M (0.0029). At these
shapes both platforms are latency-bound and far from their bounds; η is not speed
(roadmap §4.7), and times are compared only once a board gives TNCA seconds. GB10 on the
small and large tiers: 25/25 and 24/24 (Llama-3.2-1B decode 14.9 ms/token against a
9.1 ms DRAM bound).

TNCA in seconds (U280, `core-vrf-sync-v1` at 37.5 MHz, 2026-10-08): tiny + small 52/52 pass,
E 0.0072 / A 0.021 against GB10's 0.0051 / 0.041 on the same 52 points; the time ratio
T_GPU/T_TNCA is 2×10⁻⁵ to 6×10⁻³ per point ([r38_board_bringup.md](r38_board_bringup.md) §5).

## Changes to the product made by Stage 1 so far

- `rtl/tnca_shell.v`: counter read port `perf_sel`/`perf_rdata` (generated mux from
  `spec/r38/counters.json`), two data-port beat counters, and **a fix**: `done_retired`
  reported `perf_issued` since reset, so the second program of a grid claimed the first
  program's instructions too. It now reports this command's instructions, like
  `done_cycles`. Found by the S1-02 retired identity on `batched_matmul` and `streamk_matmul`.
- Transport 1.1: op 7 `PERF_READ`; a device without it answers REFUSED and the runtime
  reports `counters: {'available': False}`, never zero.
- `Session(counters=True)`, `Event.program_cycles`, `Event.counters`.
- S1-07: transposed tensor-DMA write (`HAS_TENSOR_XPOSE`, profile `core-xpose-v1`,
  capability `tensor_transpose`), the compiler lowering that uses it, and the loader
  rule `profiles.check_executes` (below).

## Measurements (O2, `core-sim`, serial-grid)

Strict η = L/T, L from `tools/r38/archbench/bounds.py` (FP lanes = vector FPU + matrix
unit; 8 B/cycle shared memory port).

| case | cycles | L | η | IPC | stall % | hazard % | LSU % | bytes moved / required |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax n=37 | 1114 | 37 | 0.033 | 0.31 | 59.7 | 36.7 | 2.9 | 1.7 |
| layernorm n=37 | 673 | 74 | 0.110 | 0.36 | 49.9 | 31.7 | 11.7 | 1.4 |
| transpose 16×16 | 4382 | 256 | 0.058 | 0.40 | 55.2 | 22.2 | 30.2 | 7.7 |
| GEMM fp16 16×16×32 | 13549 | 320 | 0.024 | 0.44 | 41.2 | 15.5 | 21.8 | 9.9 |
| GEMM int8 16×16×64 | 36718 | 288 | 0.008 | 0.31 | 57.7 | 25.4 | 29.1 | 46.0 |
| conv2d 3×4, 3×3 | 29181 | 13.5 | 0.0005 | 0.44 | 46.9 | 19.3 | 25.8 | 731 |

Dev-suite GEMMs stay at η ≈ 0.016–0.025 up to 64×64×32. The matrix unit is ≤ 0.5 % busy
even in GEMM; time goes to register hazards and single-port traffic 7.7–731× the
semantic minimum.

## S1-07: transposed tensor-DMA write (implemented)

ISA §9E specifies `t.ld.async` `mod[5]` (transposed write into an RF window).
`rtl/core/tnca_tensor.v` does not implement it, so commit `a5b53f0` gated the compiler's
`dma_load(transpose=True)` behind a capability that is false on every profile, and
`tl.trans` / `b.T` read the source down its columns through the LSU — consistent with
the transpose and GEMM traffic above.

Result (O2, serial-grid, `core-sim` → `core-xpose-v1`):

| | cycles | memory beats | LSU stall cycles | issued |
|---|---:|---:|---:|---:|
| GEMM fp16 16×16×32 | 13549 → 9249 (1.46×) | 3156 → 1616 | 2954 → 1185 | 5976 → 4730 |
| dev GEMM 32×32×64 | 90084 → 55684 (1.62×) | | | |
| stream-K stages | 23391/8668/17336/17766 → 21241/6518/13036/13466 | | | |

Cost on `tnca_tensor` (structural proxy): +4734 gates (+11.2 %), +1089 flops
(+12.5 %), +3 depth levels. With the feature off the engine is the parent's
(42347 vs 42373 gates, 8708 vs 8704 flops). A first version that accepted 1/2/4-byte
elements cost +120 % gates; the design was narrowed to the 16-bit MMA operand types.

Also closed on the way: `mod[5]` without the feature and any non-zero `mod[4:3]` now
raise ILLEGAL_INSTRUCTION instead of silently transferring the tile untransposed
(doc/r37_isa_gaps.md §1).

Known limits: the untransposed A operand of these GEMMs still goes through the LSU —
host-descriptor loads never use the transfer engine (`frontend.py`, `d.address is not
None`), which is a compiler-only opportunity and the natural S1-08 ablation arm. A
descriptor the transposed transfer cannot move (row stride not a multiple of 8 bytes)
traps DESCRIPTOR_FAULT on `core-xpose-v1` where `core-sim` succeeds via the LSU.

## Environment notes

- Icarus simulates ~400–550 target cycles per wall second on these benches; that bounds
  L1/L2 latency (smoke 836 s, dev 51 min). Verilator is not installed.
- `r36-build-tests` R36-BUILD-02 needs the pinned Triton 3.8.0, which only installs on
  x86_64 Linux; it was run on the acceptance-platform host (Ubuntu 24.04, inf2.xlarge)
  and passes there with all R38 changes.
- Four orphaned `tnca_device_vrf.vvp` relay simulations (started 03:18–04:16, parent
  PID 1) occupied four cores throughout; they were not started by this work and were
  left running.
