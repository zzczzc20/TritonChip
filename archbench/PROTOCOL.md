# ArchBench Measurement and Scoring Protocol

Version 1, 2026-10-07. Parent documents: [R38 roadmap §4 and §11](../doc/r38_isca_roadmap.md),
[Stage 1 task S1-10](../doc/r38_stage1_tasks_and_acceptance.md). This document is
also the GPU scoring protocol that S1-10 requires.

## 1. Scope

Workloads are defined at the PyTorch level. A *point* is a PyTorch module, its
inputs, a dtype, a numerical tolerance and a semantic demand. All platforms run
the same point object, built from the same seed, and each compiles it with
`torch.compile` through Inductor.

ArchBench complements TileBench, which remains the source of Triton-level
compiler and ISA coverage and of the Stage 1 TileBench scoring. Both use the
same bound formulas and scoring library (`tools/r38/archbench`).

### 1.1 Points, tiers, tracks

- Identifier: `<kind>.<family>/<tier>/<dtype>-<parameters>`, for example
  `op.matmul/tiny/fp16-m16-k32-n16` or `model.llama_decode/small/fp16-b1-p128`.
- Definitions are frozen in `suites/<tier>.json` as SHA-256 hashes of the
  parameters, dtype, tolerance, demand and track. `python -m archbench.suite
  --check` fails if the registry changes silently, and every record carries
  `point_sha256`.
- Tiers state where a point can be measured. They carry no weight.
  - `tiny`: within reach of the RTL simulator.
  - `small`: within the board's device memory; every kernel's argument
    storages fit the 384 KiB heap of the U50 image's 512 KiB memory.
  - `large`: GPU-realistic shapes. TNCA records `capacity-exceeded` until it has
    external memory; problems are never shrunk to fit.
- Tracks: `llm` covers decoder inference; `general` covers convolution, scan,
  transpose and the ResNet block.

## 2. Correctness

- **Reference.** Eager PyTorch in binary32 on the same point, with low-precision
  weights and inputs upcast exactly. `tiny` and `small` run on the CPU; `large`
  runs on CUDA with TF32 disabled.
- **Criterion.** `max|got − ref| / max(max|ref|, 1) ≤ tol`. The tolerance is
  2·10⁻⁴ for FP32, 2·10⁻² for FP16, 6·10⁻² for BF16 and 5·10⁻² for FP16
  models. Integer outputs (argmax, gathered embedding rows) must be exact, and
  NaN positions must agree.
- **Host fallback.** Any computation left on the host makes the point
  `unsupported` on that platform, even when the numbers are right.
  `tnca.inductor.host_fallbacks` scans Inductor's generated wrapper for extern
  kernels, ATen calls and C++ kernels.

## 3. Timing

### 3.1 TNCA

- **Device time** is the sum, over every kernel launch in one call, of the
  launch's region of interest (`launch_accept_to_core_idle`) in device cycles.
  Programs of a grid execute one after another (`serial-grid`), so this is not
  the makespan of a concurrent device.
- **Host transfers** are reported separately and are not device time. In Stage 1
  every launch uploads and downloads its argument storages.
- **Backend** is selected by `TNCA_BACKEND`. `sim-file` and `fpga-u50-sim`
  produce `measured-rtl` records; `fpga-u50` produces `measured-board` records.
- **Seconds.** Cycles are converted to seconds only with a measured board
  clock: the U50 image runs the device at 150 MHz / 4 = 37.5 MHz
  (`flow/fpga/README.md`). A simulated cycle has no duration.
- A TNCA launch is deterministic, and one run is the measurement. The first
  call compiles and runs.

### 3.2 GPU

- **Device time** (primary) is the sum of the durations of every device
  activity (kernels, memsets, copies) in one call, from CUPTI via
  `torch.profiler`. It is the median of *N* calls. Like the TNCA figure, it
  excludes host gaps between launches.
- **Cold cache.** Before every timed call, the caches between the SMs and
  DRAM (L2 and the system-level cache) are overwritten with a buffer twice
  their combined size, so required reads really come from DRAM. Writes, by
  contrast, can remain in a write-back cache when the kernel ends; the bound
  therefore counts only the writes that exceed the write-back capacity (§4).
  Warm-cache figures are reported and never enter η.
- **Latency.** CUDA-event latency around the call, including gaps where the
  GPU waits for the host, is reported as end-to-end device-side latency and
  never enters η.
- **Variants.** All variants are reported. The primary variant is fixed in
  advance:

  | Variant | Configuration |
  |---|---|
  | `inductor-cudagraph` (primary) | `torch.compile(mode="reduce-overhead")`: Inductor's kernels replayed as a CUDA graph |
  | `inductor` | `torch.compile()` defaults: the path a typical user gets; GEMMs may use cuBLAS |
  | `inductor-triton` | max-autotune with Triton-only GEMM templates: the code generator TNCA receives |

- **Numerics.** FP32 points run in IEEE binary32 (`allow_tf32=False`,
  `float32_matmul_precision='highest'`), matching TNCA's FP32 dot, which uses
  exact vector FMAs. FP16/BF16 points may use tensor cores with FP32
  accumulation, matching TNCA's MMA.
- **Environment.** Each record captures the GPU model, SM count, L2 size,
  driver, CUDA version, the SM clock, temperature and power at measurement time
  (`nvidia-smi`), and the PyTorch and Triton versions. Clocks are not locked
  (no root access); the observed state is recorded as-is.

## 4. Lower bound and efficiency

`L = max(compute term, memory term)`, computed from `spec_peak` only, never
from measurements. Where a peak is uncertain the generous end is used, which
can only lower `L` and make η conservative.

- **Demand** (`registry.Demand`):
  - dense MACs, where causal attention counts only unmasked products;
  - other floating-point operations and transcendentals (an undercount is safe);
  - required bytes: each input, weight and KV-cache byte read once, each output
    byte written once; an embedding counts only the rows looked up.

  `python -m archbench.selftest` checks, against `torch.utils.flop_counter`,
  that MACs are never overcounted and that dense families match exactly; it
  checks the byte demands against the actual tensors.
- **TNCA** (cycles). The Stage 1 formula (`tools/r38/archbench/bounds.py`,
  `peaks.py`): MACs and FP operations over the combined lanes of the vector FPU
  and the matrix unit; bytes over the single 8-byte-per-cycle memory port.
- **GPU** (seconds). MACs can execute on tensor cores or SIMT lanes; other
  operations only on SIMT lanes:
  `compute = max((MAC + other) / (tensor + simt), other / simt)`, with
  `tensor = 0` for FP32 points. The DRAM term is
  `(reads + max(0, writes − write-back capacity)) / bandwidth`: an output that
  fits in L2 plus the system cache need not reach DRAM before the kernel ends.
  (Bound version 1 counted every write, and the audit caught an RMSNorm with
  η = 1.2.) GB10 rates
  and their sources are in `specs/gpu_gb10.json`: 48 SMs, a 3003 MHz maximum SM
  clock, 256-bit LPDDR5X-8533 (273.1 GB/s), and a tensor-core upper estimate
  derived from the vendor's 1 PFLOPS sparse FP4 figure.
- **Efficiency.** `η = L / T`, dimensionless and comparable across platforms.
  η > 1 is an audit failure, never clipped (`score.eta` raises `BoundViolation`).

## 5. Aggregation

- Scores use the Stage 1 scoring library (`tools/r38/archbench/score.py`): the
  primary score is the weighted harmonic mean `E = (Σ wᵢ/ηᵢ)⁻¹`, reported with
  the weighted arithmetic mean `A = Σ wᵢ ηᵢ`.
- One of two weightings is fixed in advance:
  - `balanced`: equal mass per family, uniform within a family (roadmap §4.6
    balanced track);
  - reference-time: `wᵢ = T_ref,i / Σ T_ref`, where the reference is a frozen
    set of GPU primary-variant records. `python -m archbench.report gpu.jsonl
    --freeze-weights <file>` writes the weights with their own hash, and an
    edited file is refused.
- A point that failed or was not measured sets the full-suite E to 0 and marks
  it `NOT CERTIFIED`. The supported-only value is shown beside it, as a
  diagnostic.
- Absolute performance `S = Σ T_ref / Σ T` is computed only when both sides are
  in seconds. GPU times always are; TNCA times are only when measured on a
  board. Stage 1 RTL results are compared by η, not by time.

## 6. Outcomes

Failures are recorded separately and never merged into "slow": `pass`,
`numeric-error`, `compile-failure`, `runtime-fault`, `capacity-exceeded`,
`timeout`, `unsupported` (including host fallback), `not-measured` (including
compile-only sweeps).

## 7. What these measurements do not support

- Simulated cycles do not give TNCA seconds, throughput, or a speed ratio
  against a GPU.
- Higher η does not mean faster (roadmap §4.7). A single 37.5 MHz core and a
  48-SM GPU are not comparable in absolute time. A cross-platform comparison
  starts from η and its per-family distribution; absolute times follow only
  after board measurements, reported in the layers of roadmap §11.
- The GB10 tensor-core rate is an upper estimate. If the vendor publishes an
  exact figure, update `specs/gpu_gb10.json` and bump the bound version.
