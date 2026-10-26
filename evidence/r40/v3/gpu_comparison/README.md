# R40-10 item 4: GPU comparison (TNCA U280 board vs NVIDIA GB10)

Generated 2026-10-20T00:36:43+00:00 by `tools/r40/v3_gpu_compare.py` (sha256 `8a3aa4b809ea33d9…`). Data: `report.json` (every row, pair, aggregate and failure). Requirement: doc/r40_tasks_and_acceptance.md section R40-10, acceptance 4 (retained by doc/r40_tasks_and_acceptance_v3.md change 7).

## What was run

- **TNCA**: board records, image `a88ebdf59cfbbb20…`, software tree `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`, 3 cores, clock 90000000 Hz, backend fpga-u280-multicore, profile core-vrf-sync-v5, knobs `{"TNCA_ACTIVE_CORES": "3", "TNCA_OPERATING_CLOCK_EVIDENCE": "/home/zzhouch/tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json", "TNCA_TORCH_DEVICE": "native", "TNCA_TRANSFER_POLICY": "resident"}`, host acf3019.
- **GPU**: NVIDIA GB10 (48 SMs, L2 25165824 B, cc 12.1, CUDA 13.0, driver 580.95.05), torch 2.14.0+cu130, triton 3.8.0, host gx10-6646; source tree `files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca`.
  - `python -m archbench.run --platform gpu` with `--variant inductor-cudagraph` (primary) and `--variant inductor-triton`, every point of each suite (the primary ArchBench GPU records).
  - `tools/r40/v3_gpu_boundaries.py` for the same points with variants inductor-cudagraph, inductor-triton and inductor-triton-bigsm: the timings at TNCA's boundaries.
- CUDA path and the TNCA tree: the GPU children import no tnca.inductor module (the TNCA path's Inductor configuration); Triton backend discovery imports tnca.triton_backend {compiler, driver}, and the active Triton backend is the one listed. The TNCA records are on the delivery tree files:fe1b5199... (doc/r40_v3/identity_amendment.json). It differs from files:80dd1a82... only in python/tnca/inductor.py and python/tnca/triton_backend/compiler.py. The inductor.py changes are applied by tnca.inductor.configure() or gated on the TNCA device type, and configure() is called only by the TNCA platform (archbench/platforms/tnca.py, archbench/generate.py); the CUDA path imports no tnca.inductor module (recorded per boundary record). The compiler.py change (code larger than the code store raised as OutOfResources) runs only inside the TNCA Triton backend's compile; Triton backend discovery imports that module on the CUDA path, but the active backend there is cuda. The GPU records on files:80dd1a82... therefore remain the GPU side. Modules seen: ['tnca', 'tnca.triton_backend', 'tnca.triton_backend.compiler', 'tnca.triton_backend.driver']; active backend ['cuda'].
- GPU source: ~/r40v3_gpu on the GB10 host, a copy of ~/r40_bucket_full_80dd_v2_source; archbench.provenance.software_tree() = files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca, checked before the runs and recorded in every boundary record. Only tools/r40 and the points_*.txt lists at the copy root (both outside the product tree) were added.
- TNCA inputs (read-only copies from the FPGA host acf3019; content sha256 equal to the originals), all on the delivery tree files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75, image a88ebdf5, 90 MHz: tnca/u280_tiny|small|medium.jsonl.gz = ~/tnca_runs_r40_v3x2/native/u280_*.jsonl (native77, 77/77 pass), tnca/hbm14.jsonl.gz = ~/tnca_runs_r40_v3x2/hbm14.jsonl (14/14), tnca/large24.jsonl.gz = ~/tnca_v3_recollect_x2/large24.jsonl (stage original-large24, 24/24, exit 0) -- the fe1b functional re-collection (evidence/r40/v3/launches/recollect_x2_controller.py). The 80dd inputs of the first report are kept in tnca_80dd/, not paired.
- Gated GPU campaign: campaign/campaign2.sh (2026-10-19 04:08-05:48 UTC; it first waited 03:01-04:08 while another user's vLLM sweep held the GPU) and campaign/campaign3.sh (frozen inductor-triton again with --fresh-cache). Logs: campaign/campaign2.log, campaign/campaign3.log, campaign/campaign2_waitonly_0301.log; per-attempt gate lines: raw/*.gate.jsonl.
- Superseded, kept, not paired: raw/attempt1_ungated/ is a first ungated run (02:35-02:57 UTC). Its native77 inductor-cudagraph records (77/77 pass) predate the other user's job (started 02:53:15); during HBM14 that job held ~52 GB of the unified memory and two points (op.silu/hbm, op.transpose/hbm) failed with CUDA_ERROR_OUT_OF_MEMORY at context creation, after which the run was stopped and replaced by the gated campaign.
- Compile-cache incident, kept, not paired: the first gated pass of the frozen inductor-triton variant was served Inductor FX-graph-cache entries written by inductor-triton-bigsm (the cache key does not include torch._inductor.utils.is_big_gpu) and so reported Triton GEMM templates that this variant cannot build on GB10. Those records are in raw/cache_contaminated/. The frozen inductor-triton stages were rerun with an empty Inductor and Triton cache per point (campaign3.sh). The inductor-cudagraph and inductor-triton-bigsm stages shared the account's compile cache only with runs of the identical configuration.
- raw/smoke/: harness test runs (02:24-02:31 UTC, an earlier harness revision without the module/backend fields); not paired.
- Shared-GPU gate (`tools/r40/v3_gpu_campaign.py`): one point per process, started only when nvidia-smi listed no other user's compute process (a quiet window after one was seen), polled during the run; an attempt that saw one was stopped and retried. Per stage: `archbench_inductor-cudagraph_existing.gate.jsonl`: 77 clean of 77 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `archbench_inductor-cudagraph_hbm-core.gate.jsonl`: 14 clean of 14 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `archbench_inductor-cudagraph_large.gate.jsonl`: 24 clean of 24 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `archbench_inductor-triton_existing.gate.jsonl`: 77 clean of 77 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `archbench_inductor-triton_hbm-core.gate.jsonl`: 14 clean of 14 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `archbench_inductor-triton_large.gate.jsonl`: 24 clean of 24 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `boundary_inductor-cudagraph.gate.jsonl`: 113 clean of 113 attempts, 0 stopped for contention, 0 OOM retries, waited 4005 s, never clean 0, 1 with a same-account process resident; `boundary_inductor-triton.gate.jsonl`: 113 clean of 113 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident; `boundary_inductor-triton-bigsm.gate.jsonl`: 113 clean of 113 attempts, 0 stopped for contention, 0 OOM retries, waited 0 s, never clean 0, 0 with a same-account process resident. Other users seen: lizeyu. Same-account co-residents are other jobs of the measuring account that held a CUDA context (not waited for; listed in each gate line).

## Status per suite

- **native77 (tiny, small, medium)** (77 points): TNCA {'pass': 77}; gpu inductor-cudagraph archbench: {'pass': 77}; gpu inductor-cudagraph boundary: {'pass': 77}; gpu inductor-triton archbench: {'compile-failure': 27, 'pass': 50}; gpu inductor-triton boundary: {'compile-failure': 27, 'pass': 50}; gpu inductor-triton-bigsm boundary: {'pass': 77}
- **HBM14** (14 points): TNCA {'pass': 14}; gpu inductor-cudagraph archbench: {'pass': 14}; gpu inductor-cudagraph boundary: {'pass': 14}; gpu inductor-triton archbench: {'compile-failure': 5, 'pass': 9}; gpu inductor-triton boundary: {'compile-failure': 5, 'pass': 9}; gpu inductor-triton-bigsm boundary: {'pass': 14}
- **large24** (24 points): TNCA {'pass': 24}; gpu inductor-cudagraph archbench: {'pass': 24}; gpu inductor-cudagraph boundary: {'pass': 24}; gpu inductor-triton archbench: {'compile-failure': 9, 'pass': 15}; gpu inductor-triton boundary: {'compile-failure': 9, 'pass': 15}; gpu inductor-triton-bigsm boundary: {'pass': 24}

## Boundaries

| Boundary | TNCA | GPU | Match |
|---|---|---|---|
| makespan | timings["device-graph-makespan"] median cycles / clock_hz | boundary timings["cupti-span-warm"] median | matched |
| serial-sum | timings["device-cycles-serial-grid"] median cycles / clock_hz | boundary timings["cupti-sum-warm"] median | matched |
| resident-call | timings["resident-call-latency"] median, or on the native resident path timings["request-latency"] median (0 bytes uploaded or downloaded: inputs and module resident, outputs left on the device) | boundary timings["resident-call-latency-warm"] median | matched |
| request | not in the native-path records (request-latency there is the resident call) | boundary timings["request-latency"] median | gpu-only |
| archbench-gpu-primary | (none paired) TNCA device-cycles-serial-grid is the nearest kind | archbench time.device (CUPTI activity sum, L2 + system cache flushed before the call), time.latency_cold / latency_warm (CUDA events around the call) | not-matched |

- makespan: GPU timestamps come from CUPTI with the profiler active, which can lengthen host launch gaps of the non-graph variants; TNCA endpoints are device-clock MMIO counters; TNCA: one timed sample per point (functional inventory sampling); GPU: median of 10-25 profiled calls.
- serial-sum: a GPU kernel duration is its own makespan over the SMs; a TNCA launch ROI is the launch makespan over the 3 cores.
- resident-call: TNCA: one sample per point; GPU: median of 20-50 calls; host CPUs differ (TNCA: acf3019 x86_64 over PCIe MMIO; GPU: GB10 Arm cores, unified memory).
- request: no TNCA counterpart in these records; GPU figure reported for completeness.
- archbench-gpu-primary: cold cache on the GPU, no flush on TNCA; CUDA-event latency is device-side and includes the host time before the first launch; TNCA has no event-based figure; reported unchanged as the primary ArchBench GPU record and as a cross-check of the boundary harness (cupti-sum-cold should agree with it).

## Scope

- TNCA: AMD/Xilinx Alveo U280 (xcu280), FPGA prototype image; 3 cores at 90000000 Hz; 3 memory ports; exposed HBM 4294967296 B; host acf3019 over PCIe, MMIO register protocol. Power: **not measured** (null).
  - Placed utilization per SLR (SLR0/1/2, %) from `evidence/r40/v2/convergence/attempt03/slr_util_placed_20261018.rpt.gz`: CLB LUTs [73.45, 75.24, 65.21]; CLB Registers [18.16, 16.71, 12.2]; Block RAM Tile [20.68, 22.84, 18.08]; URAM [0.0, 0.0, 0.0]; DSPs [5.94, 6.15, 5.7]. The report is from the attempt03 build that produced the image; it is bound to the image by that build directory, not by a hash inside the report.
- GPU: NVIDIA GB10, 48 SMs, max SM clock 3003 MHz, L2 24 MiB + 16 MiB system cache, 256-bit LPDDR5X at 8533 MT/s (273.1 GB/s), shared with the CPU. Clocks not locked (no root); the SM clock observed during sustained loops is in each boundary record.
  - Power (nvidia-smi power.draw, GPU as the driver reports it, not wall power): sustained-loop medians across 298 point-variant records: median 16.6 W, range 15.5–64.8 W; record-start snapshot (between points) median 6.79 W. No per-point energy is derived: small points barely load the GPU in the sustained loop.

## Geometric means per family (paired points only)

Times are geometric means over the points of a family where both sides pass; ratio = geometric mean of T_TNCA / T_GPU at that boundary (above 1: TNCA took longer). See the scope and caveats before reading a ratio.

### native77 (tiny, small, medium): TNCA device-graph-makespan vs GPU CUPTI span (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 140 µs (4) | 0.719 µs | 194 | 0.896 µs (4) | 156 | 0.888 µs (4) | 157 |
| argmax | 300 µs (3) | 1.19 µs | 253 | 1.32 µs (3) | 227 | 1.32 µs (3) | 227 |
| attention_decode | 12.4 ms (3) | 4.06 µs | 3.05e+03 | 4.16 µs (3) | 2.98e+03 | 4.15 µs (3) | 2.99e+03 |
| attention_prefill | 55.1 ms (3) | 3.42 µs | 1.61e+04 | 3.52 µs (3) | 1.57e+04 | 3.52 µs (3) | 1.57e+04 |
| bmm | 2.66 ms (3) | 1.34 µs | 1.99e+03 | — (0) | — | 1.16 µs (3) | 2.3e+03 |
| conv2d | 21.9 ms (3) | 9.23 µs | 2.37e+03 | 192 µs (3) | 114 | 202 µs (3) | 108 |
| cumsum | 291 µs (3) | 0.832 µs | 350 | 1 µs (3) | 291 | 1.01 µs (3) | 288 |
| embedding | 1.31 ms (3) | 0.843 µs | 1.56e+03 | 0.939 µs (3) | 1.4e+03 | 0.939 µs (3) | 1.4e+03 |
| gelu_tanh | 340 µs (3) | 0.852 µs | 399 | 1.04 µs (3) | 326 | 1.03 µs (3) | 329 |
| gemv | 1.03 ms (3) | 1.69 µs | 611 | — (0) | — | 1.66 µs (3) | 619 |
| gpt2_block | 93.7 ms (3) | 21.3 µs | 4.4e+03 | — (0) | — | 489 µs (3) | 192 |
| layernorm | 469 µs (3) | 1.05 µs | 447 | 1.18 µs (3) | 397 | 1.18 µs (3) | 397 |
| linear | 1.24 ms (3) | 2.66 µs | 467 | — (0) | — | 2.19 µs (3) | 568 |
| llama_decode | 366 ms (3) | 81.4 µs | 4.5e+03 | — (0) | — | 2.75 ms (3) | 133 |
| llama_prefill | 724 ms (3) | 80.9 µs | 8.95e+03 | — (0) | — | 2.49 ms (3) | 291 |
| matmul | 694 µs (6) | 2.43 µs | 285 | — (0) | — | 1.93 µs (6) | 359 |
| mlp | 11.7 ms (3) | 11.3 µs | 1.04e+03 | — (0) | — | 86.4 µs (3) | 135 |
| resnet_block | 44.9 ms (3) | 19.7 µs | 2.28e+03 | 415 µs (3) | 108 | 342 µs (3) | 131 |
| rmsnorm | 368 µs (3) | 0.935 µs | 393 | 1.07 µs (3) | 345 | 1.07 µs (3) | 345 |
| rope | 10.9 ms (3) | 0.725 µs | 1.5e+04 | 0.907 µs (3) | 1.2e+04 | 0.928 µs (3) | 1.17e+04 |
| silu | 282 µs (3) | 0.767 µs | 368 | 0.939 µs (3) | 300 | 0.939 µs (3) | 300 |
| softmax | 267 µs (4) | 0.991 µs | 270 | 1.18 µs (4) | 226 | 1.18 µs (4) | 228 |
| swiglu | 322 µs (3) | 0.757 µs | 426 | 0.939 µs (3) | 344 | 0.939 µs (3) | 344 |
| transpose | 407 µs (3) | 0.811 µs | 503 | 0.992 µs (3) | 411 | 0.992 µs (3) | 411 |
| **all** | 2.46 ms (77) | 2.53 µs | 974 | 2.36 µs (50) | 511 | 5.41 µs (77) | 455 |

### native77 (tiny, small, medium): TNCA resident call latency vs GPU resident call latency (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 5.22 ms (4) | 110 µs | 47.3 | 101 µs (4) | 51.8 | 105 µs (4) | 49.5 |
| argmax | 5.05 ms (3) | 128 µs | 39.4 | 102 µs (3) | 49.3 | 107 µs (3) | 47.2 |
| attention_decode | 16.8 ms (3) | 129 µs | 130 | 156 µs (3) | 108 | 149 µs (3) | 113 |
| attention_prefill | 60.4 ms (3) | 130 µs | 463 | 157 µs (3) | 385 | 153 µs (3) | 394 |
| bmm | 20.7 ms (3) | 133 µs | 155 | — (0) | — | 111 µs (3) | 185 |
| conv2d | 26.9 ms (3) | 147 µs | 183 | 108 µs (3) | 249 | 155 µs (3) | 173 |
| cumsum | 5.17 ms (3) | 150 µs | 34.3 | 63.3 µs (3) | 81.6 | 105 µs (3) | 49 |
| embedding | 6.78 ms (3) | 168 µs | 40.3 | 35.7 µs (3) | 190 | 99.9 µs (3) | 67.8 |
| gelu_tanh | 5.85 ms (3) | 130 µs | 44.9 | 34.3 µs (3) | 171 | 99.2 µs (3) | 59 |
| gemv | 7.44 ms (3) | 78.4 µs | 94.9 | — (0) | — | 111 µs (3) | 67.1 |
| gpt2_block | 98.3 ms (3) | 152 µs | 648 | — (0) | — | 252 µs (3) | 391 |
| layernorm | 5.53 ms (3) | 111 µs | 49.9 | 20.5 µs (3) | 269 | 112 µs (3) | 49.5 |
| linear | 8.06 ms (3) | 133 µs | 60.8 | — (0) | — | 111 µs (3) | 72.8 |
| llama_decode | 371 ms (3) | 274 µs | 1.35e+03 | — (0) | — | 850 µs (3) | 436 |
| llama_prefill | 749 ms (3) | 294 µs | 2.55e+03 | — (0) | — | 753 µs (3) | 994 |
| matmul | 7.01 ms (6) | 100 µs | 69.9 | — (0) | — | 109 µs (6) | 64.4 |
| mlp | 17 ms (3) | 89 µs | 191 | — (0) | — | 128 µs (3) | 132 |
| resnet_block | 50.4 ms (3) | 139 µs | 362 | 258 µs (3) | 195 | 175 µs (3) | 288 |
| rmsnorm | 5.83 ms (3) | 129 µs | 45.1 | 36.5 µs (3) | 160 | 111 µs (3) | 52.4 |
| rope | 15.2 ms (3) | 81.2 µs | 187 | 108 µs (3) | 141 | 107 µs (3) | 141 |
| silu | 5.61 ms (3) | 129 µs | 43.4 | 104 µs (3) | 54 | 107 µs (3) | 52.4 |
| softmax | 5.9 ms (4) | 90.1 µs | 65.5 | 19.9 µs (4) | 296 | 69.7 µs (4) | 84.7 |
| swiglu | 5.61 ms (3) | 143 µs | 39.3 | 61.2 µs (3) | 91.7 | 115 µs (3) | 48.7 |
| transpose | 5.17 ms (3) | 79.6 µs | 65 | 114 µs (3) | 45.2 | 107 µs (3) | 48.4 |
| **all** | 14.1 ms (77) | 126 µs | 112 | 71.5 µs (50) | 128 | 136 µs (77) | 104 |

TNCA column: geometric mean over the inductor-cudagraph pairs; each ratio is over its own pairs.

### HBM14: TNCA device-graph-makespan vs GPU CUPTI span (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 2.71 s (1) | 1.79 ms | 1.51e+03 | 1.76 ms (1) | 1.54e+03 | 1.76 ms (1) | 1.54e+03 |
| attention_decode | 332 ms (1) | 24.4 µs | 1.36e+04 | 25.5 µs (1) | 1.3e+04 | 26 µs (1) | 1.28e+04 |
| attention_prefill | 14.4 s (1) | 26.5 µs | 5.42e+05 | 27.5 µs (1) | 5.24e+05 | 26.9 µs (1) | 5.35e+05 |
| copy | 1.81 s (1) | 1.19 ms | 1.51e+03 | 1.18 ms (1) | 1.54e+03 | 1.19 ms (1) | 1.52e+03 |
| embedding | 3.52 s (1) | 310 µs | 1.14e+04 | 305 µs (1) | 1.16e+04 | 307 µs (1) | 1.15e+04 |
| gemv | 878 ms (1) | 561 µs | 1.57e+03 | — (0) | — | 392 µs (1) | 2.24e+03 |
| linear | 19.6 s (1) | 631 µs | 3.11e+04 | — (0) | — | 612 µs (1) | 3.21e+04 |
| llama_decode | 2.29 s (1) | 1.03 ms | 2.22e+03 | — (0) | — | 3.77 ms (1) | 606 |
| llama_prefill | 49.9 s (1) | 1.44 ms | 3.47e+04 | — (0) | — | 3.62 ms (1) | 1.38e+04 |
| matmul | 47.2 s (1) | 1.64 ms | 2.88e+04 | — (0) | — | 1.39 ms (1) | 3.41e+04 |
| rmsnorm | 1.25 s (1) | 623 µs | 2.01e+03 | 607 µs (1) | 2.06e+03 | 611 µs (1) | 2.05e+03 |
| silu | 2.69 s (1) | 1.2 ms | 2.24e+03 | 1.18 ms (1) | 2.28e+03 | 1.18 ms (1) | 2.28e+03 |
| softmax | 2.49 s (1) | 618 µs | 4.03e+03 | 602 µs (1) | 4.14e+03 | 606 µs (1) | 4.11e+03 |
| transpose | 4.31 s (1) | 2.8 ms | 1.54e+03 | 2.7 ms (1) | 1.6e+03 | 2.71 ms (1) | 1.59e+03 |
| **all** | 4.01 s (14) | 579 µs | 6.92e+03 | 431 µs (9) | 5.64e+03 | 650 µs (14) | 6.16e+03 |

### HBM14: TNCA resident call latency vs GPU resident call latency (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 2.71 s (1) | 2.29 ms | 1.18e+03 | 2.03 ms (1) | 1.34e+03 | 2.02 ms (1) | 1.34e+03 |
| attention_decode | 336 ms (1) | 126 µs | 2.66e+03 | 160 µs (1) | 2.1e+03 | 178 µs (1) | 1.89e+03 |
| attention_prefill | 14.4 s (1) | 131 µs | 1.1e+05 | 169 µs (1) | 8.51e+04 | 164 µs (1) | 8.78e+04 |
| copy | 1.81 s (1) | 1.44 ms | 1.26e+03 | 1.51 ms (1) | 1.2e+03 | 1.44 ms (1) | 1.26e+03 |
| embedding | 3.53 s (1) | 654 µs | 5.4e+03 | 444 µs (1) | 7.95e+03 | 446 µs (1) | 7.91e+03 |
| gemv | 883 ms (1) | 759 µs | 1.16e+03 | — (0) | — | 524 µs (1) | 1.69e+03 |
| linear | 19.6 s (1) | 927 µs | 2.12e+04 | — (0) | — | 757 µs (1) | 2.59e+04 |
| llama_decode | 2.29 s (1) | 1.44 ms | 1.59e+03 | — (0) | — | 1.48 ms (1) | 1.55e+03 |
| llama_prefill | 50 s (1) | 2.05 ms | 2.44e+04 | — (0) | — | 1.73 ms (1) | 2.88e+04 |
| matmul | 47.2 s (1) | 2.14 ms | 2.21e+04 | — (0) | — | 1.71 ms (1) | 2.77e+04 |
| rmsnorm | 1.26 s (1) | 1.05 ms | 1.19e+03 | 741 µs (1) | 1.7e+03 | 808 µs (1) | 1.56e+03 |
| silu | 2.7 s (1) | 1.42 ms | 1.89e+03 | 1.47 ms (1) | 1.83e+03 | 1.38 ms (1) | 1.96e+03 |
| softmax | 2.5 s (1) | 909 µs | 2.74e+03 | 751 µs (1) | 3.32e+03 | 773 µs (1) | 3.23e+03 |
| transpose | 4.31 s (1) | 3.1 ms | 1.39e+03 | 3.12 ms (1) | 1.38e+03 | 3.06 ms (1) | 1.41e+03 |
| **all** | 4.02 s (14) | 972 µs | 4.14e+03 | 769 µs (9) | 3.17e+03 | 882 µs (14) | 4.56e+03 |

TNCA column: geometric mean over the inductor-cudagraph pairs; each ratio is over its own pairs.

### large24: TNCA device-graph-makespan vs GPU CUPTI span (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 678 ms (1) | 451 µs | 1.5e+03 | 437 µs (1) | 1.55e+03 | 439 µs (1) | 1.55e+03 |
| argmax | 131 ms (1) | 52.3 µs | 2.5e+03 | 48.8 µs (1) | 2.68e+03 | 49.5 µs (1) | 2.64e+03 |
| attention_decode | 331 ms (1) | 24.4 µs | 1.36e+04 | 25.5 µs (1) | 1.3e+04 | 26 µs (1) | 1.27e+04 |
| attention_prefill | 14.4 s (1) | 26.5 µs | 5.43e+05 | 27.5 µs (1) | 5.24e+05 | 26.9 µs (1) | 5.35e+05 |
| bmm | 9.25 s (1) | 93.3 µs | 9.9e+04 | — (0) | — | 77.9 µs (1) | 1.19e+05 |
| conv2d | 5.56 s (1) | 631 µs | 8.81e+03 | 868 µs (1) | 6.41e+03 | 388 µs (1) | 1.43e+04 |
| embedding | 220 ms (1) | 3.28 µs | 6.72e+04 | 3.38 µs (1) | 6.53e+04 | 3.47 µs (1) | 6.35e+04 |
| gelu_tanh | 802 ms (1) | 303 µs | 2.65e+03 | 295 µs (1) | 2.72e+03 | 297 µs (1) | 2.7e+03 |
| gemv | 333 ms (1) | 195 µs | 1.71e+03 | — (0) | — | 123 µs (1) | 2.7e+03 |
| gpt2_block | 12.4 s (1) | 239 µs | 5.19e+04 | — (0) | — | 594 µs (1) | 2.08e+04 |
| layernorm | 53.6 ms (1) | 5.25 µs | 1.02e+04 | 5.34 µs (1) | 1e+04 | 5.31 µs (1) | 1.01e+04 |
| linear | 6.46 s (1) | 241 µs | 2.68e+04 | — (0) | — | 250 µs (1) | 2.59e+04 |
| llama_decode | 28.9 s (1) | 15.1 ms | 1.92e+03 | — (0) | — | 16.2 ms (1) | 1.79e+03 |
| llama_prefill | 640 s (1) | 21.1 ms | 3.04e+04 | — (0) | — | 22.1 ms (1) | 2.89e+04 |
| matmul | 10.2 s (2) | 454 µs | 2.24e+04 | — (0) | — | 436 µs (2) | 2.33e+04 |
| mlp | 13.3 s (1) | 1.29 ms | 1.02e+04 | — (0) | — | 551 µs (1) | 2.41e+04 |
| resnet_block | 11.8 s (1) | 1.23 ms | 9.55e+03 | 1.41 ms (1) | 8.36e+03 | 761 µs (1) | 1.54e+04 |
| rmsnorm | 39.2 ms (1) | 4.21 µs | 9.31e+03 | 4.37 µs (1) | 8.97e+03 | 4.35 µs (1) | 9.01e+03 |
| rope | 116 ms (1) | 3.68 µs | 3.14e+04 | 3.78 µs (1) | 3.06e+04 | 3.71 µs (1) | 3.12e+04 |
| silu | 673 ms (1) | 303 µs | 2.22e+03 | 297 µs (1) | 2.26e+03 | 300 µs (1) | 2.24e+03 |
| softmax | 1.24 s (1) | 310 µs | 4.01e+03 | 305 µs (1) | 4.08e+03 | 303 µs (1) | 4.11e+03 |
| swiglu | 875 ms (1) | 447 µs | 1.96e+03 | 438 µs (1) | 2e+03 | 442 µs (1) | 1.98e+03 |
| transpose | 1.07 s (1) | 688 µs | 1.56e+03 | 663 µs (1) | 1.62e+03 | 663 µs (1) | 1.62e+03 |
| **all** | 1.91 s (24) | 185 µs | 1.04e+04 | 81 µs (15) | 7.75e+03 | 174 µs (24) | 1.1e+04 |

### large24: TNCA resident call latency vs GPU resident call latency (warm)

| Family | TNCA (n) | GPU cudagraph | ratio | GPU triton (n) | ratio | GPU triton-bigsm (n) | ratio |
|---|---|---|---|---|---|---|---|
| add | 683 ms (1) | 747 µs | 913 | 569 µs (1) | 1.2e+03 | 572 µs (1) | 1.19e+03 |
| argmax | 136 ms (1) | 125 µs | 1.08e+03 | 88.8 µs (1) | 1.53e+03 | 160 µs (1) | 850 |
| attention_decode | 335 ms (1) | 126 µs | 2.66e+03 | 160 µs (1) | 2.1e+03 | 178 µs (1) | 1.89e+03 |
| attention_prefill | 14.4 s (1) | 131 µs | 1.1e+05 | 169 µs (1) | 8.51e+04 | 164 µs (1) | 8.78e+04 |
| bmm | 9.26 s (1) | 431 µs | 2.15e+04 | — (0) | — | 137 µs (1) | 6.75e+04 |
| conv2d | 5.57 s (1) | 909 µs | 6.12e+03 | 804 µs (1) | 6.92e+03 | 430 µs (1) | 1.3e+04 |
| embedding | 225 ms (1) | 132 µs | 1.71e+03 | 107 µs (1) | 2.1e+03 | 123 µs (1) | 1.83e+03 |
| gelu_tanh | 806 ms (1) | 530 µs | 1.52e+03 | 357 µs (1) | 2.26e+03 | 437 µs (1) | 1.85e+03 |
| gemv | 338 ms (1) | 432 µs | 782 | — (0) | — | 201 µs (1) | 1.68e+03 |
| gpt2_block | 12.4 s (1) | 504 µs | 2.46e+04 | — (0) | — | 484 µs (1) | 2.56e+04 |
| layernorm | 57.7 ms (1) | 133 µs | 432 | 38.8 µs (1) | 1.49e+03 | 79.6 µs (1) | 725 |
| linear | 6.46 s (1) | 478 µs | 1.35e+04 | — (0) | — | 413 µs (1) | 1.56e+04 |
| llama_decode | 28.9 s (1) | 15.9 ms | 1.82e+03 | — (0) | — | 13.3 ms (1) | 2.18e+03 |
| llama_prefill | 640 s (1) | 22.5 ms | 2.85e+04 | — (0) | — | 23.8 ms (1) | 2.69e+04 |
| matmul | 10.2 s (2) | 751 µs | 1.35e+04 | — (0) | — | 537 µs (2) | 1.89e+04 |
| mlp | 13.3 s (1) | 1.74 ms | 7.64e+03 | — (0) | — | 708 µs (1) | 1.87e+04 |
| resnet_block | 11.8 s (1) | 1.57 ms | 7.5e+03 | 1.35 ms (1) | 8.69e+03 | 793 µs (1) | 1.48e+04 |
| rmsnorm | 42.6 ms (1) | 135 µs | 315 | 31.9 µs (1) | 1.34e+03 | 80 µs (1) | 533 |
| rope | 120 ms (1) | 133 µs | 907 | 33.1 µs (1) | 3.63e+03 | 81.4 µs (1) | 1.48e+03 |
| silu | 677 ms (1) | 528 µs | 1.28e+03 | 377 µs (1) | 1.79e+03 | 376 µs (1) | 1.8e+03 |
| softmax | 1.25 s (1) | 515 µs | 2.43e+03 | 460 µs (1) | 2.72e+03 | 361 µs (1) | 3.46e+03 |
| swiglu | 879 ms (1) | 682 µs | 1.29e+03 | 576 µs (1) | 1.52e+03 | 590 µs (1) | 1.49e+03 |
| transpose | 1.08 s (1) | 899 µs | 1.2e+03 | 744 µs (1) | 1.45e+03 | 811 µs (1) | 1.33e+03 |
| **all** | 1.94 s (24) | 561 µs | 3.45e+03 | 224 µs (15) | 2.87e+03 | 412 µs (24) | 4.7e+03 |

TNCA column: geometric mean over the inductor-cudagraph pairs; each ratio is over its own pairs.

All variants and the serial-sum boundary are in `report.json` (`family_geomeans`).

## Per point: HBM14

Makespan: TNCA device-graph-makespan / GPU CUPTI span (warm). Resident: host resident call latency. A status replaces a time where that side did not pass.

| Point | TNCA makespan | GPU cudagraph | GPU triton | GPU triton-bigsm | TNCA resident | GPU cudagraph resident | GPU triton-bigsm resident | TNCA gap fraction |
|---|---|---|---|---|---|---|---|---|
| `model.llama_decode/hbm/fp16-b1-p512` | 2.29 s | 1.03 ms | compile-failure | 3.77 ms | 2.29 s | 1.44 ms | 1.48 ms | 0.172 |
| `model.llama_prefill/hbm/fp16-b1-s512` | 49.9 s | 1.44 ms | compile-failure | 3.62 ms | 50 s | 2.05 ms | 1.73 ms | 0.014 |
| `op.add/hbm/fp16-n67108864` | 2.71 s | 1.79 ms | 1.76 ms | 1.76 ms | 2.71 s | 2.29 ms | 2.02 ms | 0.000 |
| `op.attention_decode/large/fp16-b1-h32-l2048-d64` | 332 ms | 24.4 µs | 25.5 µs | 26 µs | 336 ms | 126 µs | 178 µs | 0.023 |
| `op.attention_prefill/large/fp16-b1-h32-s512-d64` | 14.4 s | 26.5 µs | 27.5 µs | 26.9 µs | 14.4 s | 131 µs | 164 µs | 0.003 |
| `op.copy/hbm/fp16-n67108864` | 1.81 s | 1.19 ms | 1.18 ms | 1.19 ms | 1.81 s | 1.44 ms | 1.44 ms | 0.000 |
| `op.embedding/hbm/fp16-v65536-d2048-t8192` | 3.52 s | 310 µs | 305 µs | 307 µs | 3.53 s | 654 µs | 446 µs | 0.000 |
| `op.gemv/hbm/fp16-k4096-n11008` | 878 ms | 561 µs | compile-failure | 392 µs | 883 ms | 759 µs | 524 µs | 0.000 |
| `op.linear/hbm/fp16-m512-k4096-n11008` | 19.6 s | 631 µs | compile-failure | 612 µs | 19.6 s | 927 µs | 757 µs | 0.000 |
| `op.matmul/hbm/fp16-m4096-k4096-n4096` | 47.2 s | 1.64 ms | compile-failure | 1.39 ms | 47.2 s | 2.14 ms | 1.71 ms | 0.000 |
| `op.rmsnorm/hbm/fp16-r8192-d4096` | 1.25 s | 623 µs | 607 µs | 611 µs | 1.26 s | 1.05 ms | 808 µs | 0.000 |
| `op.silu/hbm/fp16-n67108864` | 2.69 s | 1.2 ms | 1.18 ms | 1.18 ms | 2.7 s | 1.42 ms | 1.38 ms | 0.000 |
| `op.softmax/hbm/fp16-r4096-c8192` | 2.49 s | 618 µs | 602 µs | 606 µs | 2.5 s | 909 µs | 773 µs | 0.000 |
| `op.transpose/hbm/fp32-r8192-c8192` | 4.31 s | 2.8 ms | 2.7 ms | 2.71 ms | 4.31 s | 3.1 ms | 3.06 ms | 0.000 |

## Per point: large24

Makespan: TNCA device-graph-makespan / GPU CUPTI span (warm). Resident: host resident call latency. A status replaces a time where that side did not pass.

| Point | TNCA makespan | GPU cudagraph | GPU triton | GPU triton-bigsm | TNCA resident | GPU cudagraph resident | GPU triton-bigsm resident | TNCA gap fraction |
|---|---|---|---|---|---|---|---|---|
| `model.gpt2_block/large/fp16-d1024-h16-b1-s512` | 12.4 s | 239 µs | compile-failure | 594 µs | 12.4 s | 504 µs | 484 µs | 0.005 |
| `model.llama_decode/large/fp16-b1-p512` | 28.9 s | 15.1 ms | compile-failure | 16.2 ms | 28.9 s | 15.9 ms | 13.3 ms | 0.057 |
| `model.llama_prefill/large/fp16-b1-s512` | 640 s | 21.1 ms | compile-failure | 22.1 ms | 640 s | 22.5 ms | 23.8 ms | 0.003 |
| `model.mlp/large/fp16-t512-d2048` | 13.3 s | 1.29 ms | compile-failure | 551 µs | 13.3 s | 1.74 ms | 708 µs | 0.000 |
| `model.resnet_block/large/fp32-n8-c64-hw56` | 11.8 s | 1.23 ms | 1.41 ms | 761 µs | 11.8 s | 1.57 ms | 793 µs | 0.003 |
| `op.add/large/fp16-n16777216` | 678 ms | 451 µs | 437 µs | 439 µs | 683 ms | 747 µs | 572 µs | 0.000 |
| `op.argmax/large/fp16-r64-v128256` | 131 ms | 52.3 µs | 48.8 µs | 49.5 µs | 136 ms | 125 µs | 160 µs | 0.000 |
| `op.attention_decode/large/fp16-b1-h32-l2048-d64` | 331 ms | 24.4 µs | 25.5 µs | 26 µs | 335 ms | 126 µs | 178 µs | 0.021 |
| `op.attention_prefill/large/fp16-b1-h32-s512-d64` | 14.4 s | 26.5 µs | 27.5 µs | 26.9 µs | 14.4 s | 131 µs | 164 µs | 0.003 |
| `op.bmm/large/fp16-b32-m512-k64-n512` | 9.25 s | 93.3 µs | compile-failure | 77.9 µs | 9.26 s | 431 µs | 137 µs | 0.000 |
| `op.conv2d/large/fp32-n8-cin64-h56-w56-cout64` | 5.56 s | 631 µs | 868 µs | 388 µs | 5.57 s | 909 µs | 430 µs | 0.002 |
| `op.embedding/large/fp16-v128256-d2048-t512` | 220 ms | 3.28 µs | 3.38 µs | 3.47 µs | 225 ms | 132 µs | 123 µs | 0.000 |
| `op.gelu_tanh/large/fp16-n16777216` | 802 ms | 303 µs | 295 µs | 297 µs | 806 ms | 530 µs | 437 µs | 0.000 |
| `op.gemv/large/fp16-k2048-n8192` | 333 ms | 195 µs | compile-failure | 123 µs | 338 ms | 432 µs | 201 µs | 0.000 |
| `op.layernorm/large/fp16-r512-d2048` | 53.6 ms | 5.25 µs | 5.34 µs | 5.31 µs | 57.7 ms | 133 µs | 79.6 µs | 0.000 |
| `op.linear/large/fp16-m512-k2048-n8192` | 6.46 s | 241 µs | compile-failure | 250 µs | 6.46 s | 478 µs | 413 µs | 0.000 |
| `op.matmul/large/fp16-m2048-k2048-n2048` | 5.94 s | 193 µs | compile-failure | 195 µs | 5.94 s | 383 µs | 255 µs | 0.000 |
| `op.matmul/large/fp32-m2048-k2048-n2048` | 17.4 s | 1.07 ms | compile-failure | 977 µs | 17.4 s | 1.47 ms | 1.13 ms | 0.000 |
| `op.rmsnorm/large/fp16-r512-d2048` | 39.2 ms | 4.21 µs | 4.37 µs | 4.35 µs | 42.6 ms | 135 µs | 80 µs | 0.000 |
| `op.rope/large/fp16-b1-h32-s512-d64` | 116 ms | 3.68 µs | 3.78 µs | 3.71 µs | 120 ms | 133 µs | 81.4 µs | 0.092 |
| `op.silu/large/fp16-n16777216` | 673 ms | 303 µs | 297 µs | 300 µs | 677 ms | 528 µs | 376 µs | 0.000 |
| `op.softmax/large/fp16-r4096-c4096` | 1.24 s | 310 µs | 305 µs | 303 µs | 1.25 s | 515 µs | 361 µs | 0.000 |
| `op.swiglu/large/fp16-n16777216` | 875 ms | 447 µs | 438 µs | 442 µs | 879 ms | 682 µs | 590 µs | 0.000 |
| `op.transpose/large/fp32-r4096-c4096` | 1.07 s | 688 µs | 663 µs | 663 µs | 1.08 s | 899 µs | 811 µs | 0.000 |

## Failures (kept, not dropped)

- gpu inductor-triton archbench, **compile-failure** (41): NoValidChoicesError: No choices to select
  - model.gpt2_block/medium/fp16-d160-h5-b1-s64 (existing), model.gpt2_block/small/fp16-d128-h4-b1-s64 (existing), model.gpt2_block/tiny/fp16-d64-h4-b1-s16 (existing), model.llama_decode/medium/fp16-b1-p256 (existing), model.llama_decode/small/fp16-b1-p128 (existing), model.llama_decode/tiny/fp16-b1-p16 (existing), model.llama_prefill/medium/fp16-b1-s64 (existing), model.llama_prefill/small/fp16-b1-s64 (existing), model.llama_prefill/tiny/fp16-b1-s16 (existing), model.mlp/medium/fp16-t128-d128 (existing), model.mlp/small/fp16-t64-d128 (existing), model.mlp/tiny/fp16-t16-d64 (existing), op.bmm/medium/fp16-b16-m64-k32-n64 (existing), op.bmm/small/fp16-b8-m64-k32-n64 (existing), op.bmm/tiny/fp16-b2-m16-k16-n16 (existing), op.gemv/medium/fp16-k320-n448 (existing), op.gemv/small/fp16-k256-n512 (existing), op.gemv/tiny/fp16-k64-n64 (existing), op.linear/medium/fp16-m128-k320-n256 (existing), op.linear/small/fp16-m64-k256-n256 (existing), op.linear/tiny/fp16-m16-k64-n64 (existing), op.matmul/medium/fp16-m128-k192-n128 (existing), op.matmul/medium/fp32-m128-k192-n128 (existing), op.matmul/small/fp16-m64-k128-n128 (existing), op.matmul/small/fp32-m64-k128-n128 (existing), op.matmul/tiny/fp16-m16-k32-n16 (existing), op.matmul/tiny/fp32-m16-k32-n16 (existing), model.llama_decode/hbm/fp16-b1-p512 (hbm-core), model.llama_prefill/hbm/fp16-b1-s512 (hbm-core), op.gemv/hbm/fp16-k4096-n11008 (hbm-core), op.linear/hbm/fp16-m512-k4096-n11008 (hbm-core), op.matmul/hbm/fp16-m4096-k4096-n4096 (hbm-core), model.gpt2_block/large/fp16-d1024-h16-b1-s512 (large), model.llama_decode/large/fp16-b1-p512 (large), model.llama_prefill/large/fp16-b1-s512 (large), model.mlp/large/fp16-t512-d2048 (large), op.bmm/large/fp16-b32-m512-k64-n512 (large), op.gemv/large/fp16-k2048-n8192 (large), op.linear/large/fp16-m512-k2048-n8192 (large), op.matmul/large/fp16-m2048-k2048-n2048 (large), op.matmul/large/fp32-m2048-k2048-n2048 (large)
- gpu inductor-triton boundary, **compile-failure** (41): NoValidChoicesError: No choices to select
  - model.gpt2_block/medium/fp16-d160-h5-b1-s64 (existing), model.gpt2_block/small/fp16-d128-h4-b1-s64 (existing), model.gpt2_block/tiny/fp16-d64-h4-b1-s16 (existing), model.llama_decode/medium/fp16-b1-p256 (existing), model.llama_decode/small/fp16-b1-p128 (existing), model.llama_decode/tiny/fp16-b1-p16 (existing), model.llama_prefill/medium/fp16-b1-s64 (existing), model.llama_prefill/small/fp16-b1-s64 (existing), model.llama_prefill/tiny/fp16-b1-s16 (existing), model.mlp/medium/fp16-t128-d128 (existing), model.mlp/small/fp16-t64-d128 (existing), model.mlp/tiny/fp16-t16-d64 (existing), op.bmm/medium/fp16-b16-m64-k32-n64 (existing), op.bmm/small/fp16-b8-m64-k32-n64 (existing), op.bmm/tiny/fp16-b2-m16-k16-n16 (existing), op.gemv/medium/fp16-k320-n448 (existing), op.gemv/small/fp16-k256-n512 (existing), op.gemv/tiny/fp16-k64-n64 (existing), op.linear/medium/fp16-m128-k320-n256 (existing), op.linear/small/fp16-m64-k256-n256 (existing), op.linear/tiny/fp16-m16-k64-n64 (existing), op.matmul/medium/fp16-m128-k192-n128 (existing), op.matmul/medium/fp32-m128-k192-n128 (existing), op.matmul/small/fp16-m64-k128-n128 (existing), op.matmul/small/fp32-m64-k128-n128 (existing), op.matmul/tiny/fp16-m16-k32-n16 (existing), op.matmul/tiny/fp32-m16-k32-n16 (existing), model.llama_decode/hbm/fp16-b1-p512 (hbm-core), model.llama_prefill/hbm/fp16-b1-s512 (hbm-core), op.gemv/hbm/fp16-k4096-n11008 (hbm-core), op.linear/hbm/fp16-m512-k4096-n11008 (hbm-core), op.matmul/hbm/fp16-m4096-k4096-n4096 (hbm-core), model.gpt2_block/large/fp16-d1024-h16-b1-s512 (large), model.llama_decode/large/fp16-b1-p512 (large), model.llama_prefill/large/fp16-b1-s512 (large), model.mlp/large/fp16-t512-d2048 (large), op.bmm/large/fp16-b32-m512-k64-n512 (large), op.gemv/large/fp16-k2048-n8192 (large), op.linear/large/fp16-m512-k2048-n8192 (large), op.matmul/large/fp16-m2048-k2048-n2048 (large), op.matmul/large/fp32-m2048-k2048-n2048 (large)

## Cross-checks

- inductor-cudagraph: boundary cupti-sum-cold / archbench time.device: n=115, geomean 0.998, median 1.000, range 0.765–1.456
- inductor-triton: boundary cupti-sum-cold / archbench time.device: n=74, geomean 0.983, median 0.990, range 0.819–1.169
- points measured in two suites (archbench time.device): op.attention_decode/large/fp16-b1-h32-l2048-d64 inductor-cudagraph: hbm-core 123 µs, large 126 µs
- points measured in two suites (archbench time.device): op.attention_decode/large/fp16-b1-h32-l2048-d64 inductor-triton: hbm-core 130 µs, large 122 µs
- points measured in two suites (archbench time.device): op.attention_prefill/large/fp16-b1-h32-s512-d64 inductor-cudagraph: hbm-core 73.2 µs, large 73.9 µs
- points measured in two suites (archbench time.device): op.attention_prefill/large/fp16-b1-h32-s512-d64 inductor-triton: hbm-core 75.6 µs, large 72 µs

## Caveats

- Different resource and power scope: TNCA is a 3-core FPGA prototype at 90 MHz on a U280 card (an FPGA-fabric-limited clock); the GPU is a 48-SM GB10 at up to 3 GHz sharing LPDDR5X with its CPU. Time ratios here compare these two systems as built; they are not normalised by area, clock, power or process and do not predict an ASIC.
- No commercial-ASIC figures are used or compared (R40-10 acceptance 4).
- TNCA sampling is the functional inventory plan: one untimed warm-up call and one timed call per point. Device-cycle kinds are deterministic within 0.2 % on this board (v3 change 5); the host resident-call latency is a single sample and carries host noise.
- Warm vs cold: TNCA timed calls follow one warm-up with no cache flush, so the paired GPU figures are the warm (no-flush) ones. ArchBench's primary GPU device time is cold-L2 and is reported beside them, not paired.
- The TNCA makespan includes host submission gaps between launches (MMIO dispatch per launch); gap_fraction = 1 - serial-sum/makespan shows how much of each makespan they are. Under inductor-cudagraph the GPU replays the whole call as one CUDA graph; under the Triton variants kernels are launched one by one from Python and their host gaps enter the GPU span the same way.
- Precision is the point's dtype on both sides (FP16 with FP32 accumulation on tensor cores / MMA; FP32 points in IEEE binary32, TF32 off) and both are checked against the same FP32 reference and tolerance.
- inductor-triton as ArchBench freezes it (max_autotune_gemm with backends=TRITON) has no GEMM choice on GB10: Inductor disables Triton GEMM templates below 68 SMs (is_big_gpu), so its GEMM points fail with NoValidChoicesError. These failures are kept. inductor-triton-bigsm is the same configuration with that check forced true, a supplementary variant from tools/r40/v3_gpu_boundaries.py, not an ArchBench variant.
- "Triton-only" covers GEMMs: in the Triton variants Inductor still lowers some operators to library kernels (e.g. SDPA to flash attention, convolution to cuDNN); kernel_classes and triton_only in each boundary row list what ran.
- GPU clocks are not locked and the GPU is shared; the SM clock and power during the sustained loop and the nvidia-smi state at each record start are kept in the raw records.
- GPU kernels of the small tiers run for about 1-10 µs; the same point measured in separate processes (ArchBench cold sum vs the boundary harness cold sum) differs by a factor 0.77-1.46, so per-point GPU figures at that scale carry that spread.
- Software identity differs between sides: TNCA records ['files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75'], GPU runs ['files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca']. The TNCA records are on the delivery tree files:fe1b5199... (doc/r40_v3/identity_amendment.json). It differs from files:80dd1a82... only in python/tnca/inductor.py and python/tnca/triton_backend/compiler.py. The inductor.py changes are applied by tnca.inductor.configure() or gated on the TNCA device type, and configure() is called only by the TNCA platform (archbench/platforms/tnca.py, archbench/generate.py); the CUDA path imports no tnca.inductor module (recorded per boundary record). The compiler.py change (code larger than the code store raised as OutOfResources) runs only inside the TNCA Triton backend's compile; Triton backend discovery imports that module on the CUDA path, but the active backend there is cuda. The GPU records on files:80dd1a82... therefore remain the GPU side.

## Files

| File | sha256 | Records |
|---|---|---|
| `evidence/r40/v3/gpu_comparison/tnca/u280_tiny.jsonl.gz` | `fa3e53abbea69adb70232173a24fd5552739c46545172977ef2df4016baa3069` (content `99e15d77ed1c4996f2dc715001532138b57ba6669abd77cb70592e99ed6469f8`) | 27 |
| `evidence/r40/v3/gpu_comparison/tnca/u280_small.jsonl.gz` | `7abab420b9c67a48912108982e27456fc7c2f1251c5fee30e58aa8114017e5d5` (content `566b8701936d2b66d167d38315c1f6f8a1a243c79b3772b758b8c1ac3e492b61`) | 25 |
| `evidence/r40/v3/gpu_comparison/tnca/u280_medium.jsonl.gz` | `8d7692deb5c65a96777b885954689924b40a58ed3b29263161e6521e614ff50d` (content `d4fcc17648081bf3da4c0fca6e1888a4509aeeab082bb2d74cd7ec1245f7d4eb`) | 25 |
| `evidence/r40/v3/gpu_comparison/tnca/hbm14.jsonl.gz` | `5513ced9912c87b7ee9bb661994bdbd7266ae9abea0ed27412e84b4b964a6af1` (content `8b27b796715092c533e3992c7cafd276fcec6a9a0d36f988b1537905a006857a`) | 14 |
| `evidence/r40/v3/gpu_comparison/tnca/large24.jsonl.gz` | `5ecf2791b1b3b663bab4a8feec0d7501738ddf4ebbfe4a9cf5380d4f6ae1f4e1` (content `977de705160fde2ee99e23cfcca30b92dce889d457371afbcab0863fde9b9131`) | 24 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_existing.jsonl` | `ae2d2e0613b2e80993c7bf9f8162ca17e094b8bb18251b0408c50c8806b952ab` | 77 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_hbm-core.jsonl` | `bfdc6011b401d950910630c4fe2825f5bd33304f686e81c4b7b4fd8a44645c05` | 14 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_large.jsonl` | `c554f23d1e882848af54f762b6877e6c26ce36e48103a02e4d0c9463cf3a4dc3` | 24 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_existing.jsonl` | `3d6377ecb9d4641b0121a2a7ee88cb8b20fce3fa3c1475bdf8553e3401654990` | 77 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_hbm-core.jsonl` | `81ac7df3357158ec75b8f9b88970ee2e9d8cc6de78675865e86cc75d10cef710` | 14 |
| `evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_large.jsonl` | `17fb9af7320028ec66550c5417a7bf7a4495667cd52aa083176863dd4d315d5e` | 24 |
| `evidence/r40/v3/gpu_comparison/raw/boundary_inductor-cudagraph.jsonl` | `8cbe44fadec5f7fc55a0c203c341707bf52353aeef26542da0b62b81eed304df` | 113 |
| `evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton.jsonl` | `4970b9a028930c7bcb18445126fc27cb55d9b43c3829c74efd6d166bead3caf6` | 113 |
| `evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton-bigsm.jsonl` | `8206edee96c2c8e14a2053a5461fbaa81c5531a4a70e23c42b9a480d18d52d55` | 113 |
| `evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_existing.jsonl` | `9e8739ab40a571548dcbdc3844048f2c738b5d7a295d19b6896d3174fbbf83a2` | |
| `evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_hbm-core.jsonl` | `1276d99aad84ac42967346378073f697b31585af28f4f91f375209ed8016ba68` | |
| `evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_large.jsonl` | `cbe0643913c636779020c2d9280f6a8dd17b527621051a5f93ad29417a4123f0` | |
| `evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/campaign.log` | `d16033e7d9a72bad6f6f66c57dcf6e1033fb507116fe173c5481b514b4e7465b` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_existing.gate.jsonl` | `a2afcb110de4195dce6149a7d5fabed2dc1996b3647616a042ee29f2a1627964` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_existing.jsonl` | `a8bfd898116f31c77d9958835119e980096e5e9895480878b27d13a742b21029` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_hbm-core.gate.jsonl` | `783d08f19e31d0fb4cea5ce94f5cf19dba1b9ff983f9b048bf203de56a0d65b2` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_hbm-core.jsonl` | `f3499bd276d04788699b04a6a668b1a9623a7c75a3280f42c0705231ae6d5de6` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_large.gate.jsonl` | `99fba9eada8e208c5282f80785fd3c8dcf1a527aa47ab0fa3aa90da34f8956cf` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_large.jsonl` | `e5a83f496d62cfdf191ef445007c759f2e670e105ace4fdf5618bd308059f8ea` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/boundary_inductor-triton.gate.jsonl` | `4c7d676e3374f0a6987d100cd8d0bd4f5890f55fb50bb7326eb9ecf889cf936a` | |
| `evidence/r40/v3/gpu_comparison/raw/cache_contaminated/boundary_inductor-triton.jsonl` | `ed9c7735297e1a6987504d42a9090d9208a44180fcbd11b021cdb66f363e3ff3` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/campaign2.log` | `d9f9e1e9930a66ffbd3e87be6b4379522dbfd866707ea9faa99f18a3b4dd4c34` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/campaign2.sh` | `034105bb8a5d33784774f4c5c93600ad7766f63eceedca4ba100b34793a055f6` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/campaign2_waitonly_0301.log` | `eb712e973291231c247f6b8a8c76813e70ace66be99aa76224648775517a9e9e` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/campaign3.log` | `ecc94dba7b772679d8554aeef2544758cb672c77029793295edaa1e3f90e2cac` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/campaign3.sh` | `9a4f9c5835519a42e67bad94bda5b036bce51e29f2239da85c23bd3abd08f0b5` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/points_unique.txt` | `012a75cc6651c3627792cfd5d1c649f7d2dda4472bb835189088e01d62c19489` | |
| `evidence/r40/v3/gpu_comparison/raw/smoke/bd_bigsm.jsonl` | `fbe8a9944d7efc0b0931c9190d33a6d5a20c007b766064bcfbfafc09cb9f5249` | |
| `evidence/r40/v3/gpu_comparison/raw/smoke/bd_cg.jsonl` | `4788187a09ac4c93a0013e604ede13701e1447740c6b9f37200c69a8463177f4` | |
| `evidence/r40/v3/gpu_comparison/raw/smoke/bd_mods.jsonl` | `b0778d846178370dbc1163bf7a7af8b3dd3511a3e48e462d719808cd8bb831f9` | |
| `evidence/r40/v3/gpu_comparison/raw/smoke/cg.jsonl` | `9a74933c8a2fd8e4360c34cd2066f611bb8c30c20cba04fd5f9d039f19289ec5` | |
| `evidence/r40/v3/gpu_comparison/raw/smoke/tr.jsonl` | `45b135802d0b1a3c097bf8053f4400c7389d1ade4b6848b0d1c330cd598619a5` | |
| `evidence/r40/v3/gpu_comparison/raw/campaign/notes.json` | `32b2f09869994c3652850c3415af73df0397e9ec50e9da35323be72523ade426` | |
| `evidence/r40/v2/convergence/attempt03/slr_util_placed_20261018.rpt.gz` | `20c0ee876b0bf000d7f0fed693558540b9465654c2f056af5ec5cecb8d674da2` | |

Gzipped inputs: the first hash is of the .gz file, the content hash of the JSONL inside.

Regenerate (from the repository root):

```sh
python3 tools/r40/v3_gpu_compare.py --tnca evidence/r40/v3/gpu_comparison/tnca/u280_tiny.jsonl.gz evidence/r40/v3/gpu_comparison/tnca/u280_small.jsonl.gz evidence/r40/v3/gpu_comparison/tnca/u280_medium.jsonl.gz evidence/r40/v3/gpu_comparison/tnca/hbm14.jsonl.gz evidence/r40/v3/gpu_comparison/tnca/large24.jsonl.gz --gpu-archbench evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_existing.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_hbm-core.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_large.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_existing.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_hbm-core.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_large.jsonl --gpu-boundary evidence/r40/v3/gpu_comparison/raw/boundary_inductor-cudagraph.jsonl evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton.jsonl evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton-bigsm.jsonl --gpu-gate evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_existing.gate.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_hbm-core.gate.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-cudagraph_large.gate.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_existing.gate.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_hbm-core.gate.jsonl evidence/r40/v3/gpu_comparison/raw/archbench_inductor-triton_large.gate.jsonl evidence/r40/v3/gpu_comparison/raw/boundary_inductor-cudagraph.gate.jsonl evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton.gate.jsonl evidence/r40/v3/gpu_comparison/raw/boundary_inductor-triton-bigsm.gate.jsonl --tnca-resources evidence/r40/v2/convergence/attempt03/slr_util_placed_20261018.rpt.gz --notes-file evidence/r40/v3/gpu_comparison/raw/campaign/notes.json --extra-file evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_existing.jsonl evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_hbm-core.jsonl evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/archbench_inductor-cudagraph_large.jsonl evidence/r40/v3/gpu_comparison/raw/attempt1_ungated/campaign.log evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_existing.gate.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_existing.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_hbm-core.gate.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_hbm-core.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_large.gate.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/archbench_inductor-triton_large.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/boundary_inductor-triton.gate.jsonl evidence/r40/v3/gpu_comparison/raw/cache_contaminated/boundary_inductor-triton.jsonl evidence/r40/v3/gpu_comparison/raw/campaign/campaign2.log evidence/r40/v3/gpu_comparison/raw/campaign/campaign2.sh evidence/r40/v3/gpu_comparison/raw/campaign/campaign2_waitonly_0301.log evidence/r40/v3/gpu_comparison/raw/campaign/campaign3.log evidence/r40/v3/gpu_comparison/raw/campaign/campaign3.sh evidence/r40/v3/gpu_comparison/raw/campaign/points_unique.txt evidence/r40/v3/gpu_comparison/raw/smoke/bd_bigsm.jsonl evidence/r40/v3/gpu_comparison/raw/smoke/bd_cg.jsonl evidence/r40/v3/gpu_comparison/raw/smoke/bd_mods.jsonl evidence/r40/v3/gpu_comparison/raw/smoke/cg.jsonl evidence/r40/v3/gpu_comparison/raw/smoke/tr.jsonl --out-dir evidence/r40/v3/gpu_comparison
```
