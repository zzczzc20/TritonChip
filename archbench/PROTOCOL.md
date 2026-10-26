# ArchBench Measurement and Scoring Protocol

Version 2, 2026-10-10 (R40-00). Parent documents: [R38 roadmap §4 and §11](../doc/r38_isca_roadmap.md),
[Stage 1 task S1-10](../doc/r38_stage1_tasks_and_acceptance.md),
[R40 tasks and acceptance §3](../doc/r40_tasks_and_acceptance.md). This document is
also the GPU scoring protocol that S1-10 requires.

Version 2 adds timing kinds, timed samples and provenance to TNCA records
(§3.1–3.3, §5.1); records say which they follow (`schema`:
`archbench-torch-record/1` or `/2`). Version 1 records stay valid as what they
are: the frozen R39 baseline B39 (`spec/r40/baseline.json`) is protocol 1, and
`tools/r40/b39.py wrap` gives it the provenance version 2 asks for, from the
one file that records it.

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
  - `medium` (R39-13): a moderately bigger load for the board, two to eight
    times `small`; every kernel's largest launch stays within the same heap
    with room for the runtime's own scratch (at most 352 KiB of arguments).
  - `large`: GPU-realistic shapes. TNCA records `capacity-exceeded` until it has
    external memory; problems are never shrunk to fit.
  - `hbm` (R40-00, `archbench/hbm.py`): working sets of at least 32 MiB, four
    times the architecture's whole NUCA, for streaming, GEMM, transpose and
    gather, and the R40 decoder (4 layers, 211 MiB of weights). R40's *HBM
    suite* is every point of every tier plus the multi-step decoder workload
    (`spec/r40/suites/hbm.json`, `spec/r40/plan.json`).
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

- **Device time** (`device-cycles-serial-grid`) is the sum, over every kernel
  launch in one call, of the launch's region of interest
  (`launch_accept_to_core_idle`) in device cycles. Programs of a grid execute
  one after another (`serial-grid`), so this is not the makespan of a
  concurrent device.
- **Who counts.** A record carries the shell's own count (`done_cycles`). Where
  the device under test may have been changed by whoever is being measured, as
  in the agent harness, device time is instead the platform's count: from the
  edge that accepts the launch to the first edge that sees its completion, kept
  by the observer in `tb/device/tnca_device.v` (`+observe=`). The two differ by
  the shell's handshake, a few cycles per launch, so scores compare only counts
  of one kind.
- **Host transfers** are reported separately and are not device time. On the
  CPU-tensor path (B39) every launch uploads and downloads its argument
  storages.
- **Backend** is selected by `TNCA_BACKEND`. `sim-file` and `fpga-u50-sim`
  produce `measured-rtl` records; `fpga-u50` and `fpga-u280` produce
  `measured-board` records.
- **Seconds.** Cycles are converted to seconds only with a measured board
  clock: the U280 images since 2026-10-09 run the device at 100 MHz,
  undivided (measured 99.99–100.02 MHz); the U50 image ran it at 37.5 MHz. A
  simulated cycle has no duration.
- **Protocol 1.** One run is the measurement: the first call compiles and
  runs, and its launches' device cycles are the time.
- **Protocol 2** (`archbench.run --warmup W --timed N`). The cold first call --
  Dynamo, Inductor, autotuning, the tnca lowering, then the launches -- is
  recorded on its own (`time.first_call`, `host.first_call_s`) and is never a
  sample. `W` untimed calls follow, then `N` timed calls, each one sample:
  - `timings["device-cycles-serial-grid"]`: the call's device cycles;
  - `timings["request-latency"]`: host monotonic time around the call. On the
    CPU-tensor path it contains every launch's upload and download, since
    the call cannot run without them (§3.3).
  - With `N = 0` (the simulator: deterministic, too slow to repeat) the first
    call's device cycles are the one sample, marked `deterministic`; there is
    no latency.
  - Every timed call is checked as the first is: the wrapper's launch count,
    nonzero device time, and the last call's outputs against the reference.
  - `host.steady` gives the median upload, run and download seconds of a timed
    call, so compilation (first call − steady call) and transfers can be
    checked against the samples from the record alone.
- **Sample plan** (`spec/r40/plan.json`, `sample_plan`): on the board, per
  point at least 5 warm-up and 30 timed calls in 3 independent blocks, A and
  B interleaved in randomised order; a point whose call exceeds 1 s, at least
  10 calls in 3 blocks. Medians, p95 and a paired bootstrap interval are
  reported.

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

### 3.3 Timing kinds (protocol 2)

A comparison uses one kind on both sides; `tools/r40/ablate.py` refuses a
pair measured in different kinds, or in a kind the experiment did not freeze.

| Kind | What it measures |
|---|---|
| `device-cycles-serial-grid` | §3.1: the sum of the call's launches, programs of a grid one after another. B39's kind |
| `kernel-makespan` | one grid, from the device accepting it until every core is done and its writes are visible in the stated domain |
| `graph-makespan` | from the graph's first submission until its last dependency completes, device waits and scheduling gaps included |
| `resident-call-latency` | weights and inputs already resident; host monotonic time from the call until it is complete |
| `request-latency` | as above, plus the explicit input copies to the device and output copies back |

- A makespan is measured end to end, on one readable time base. It is never
  the sum of per-core active cycles or of overlapping kernels' times; a
  record whose makespan equals such a sum is refused.
- Local cycles of different clock domains are recorded with their frequencies
  and converted, never added.
- Warm (resident) and cold-cache figures are reported apart. A cold-cache
  flush covers the actual NUCA capacity.
- Compilation, autotuning, weight initialisation and the warm-up of a replay
  are recorded on their own and are never inside a timed sample.

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

### 5.1 A/B comparisons (protocol 2, `tools/r40/ablate.py`)

- **Speed** is `S_geo = exp(mean_i log(T_A,i / T_B,i))` over a subset frozen
  before either side was measured (`spec/r40/suites/`), `T` the median of a
  point's timed samples (R39's definition). The balanced-weighted geometric
  mean is reported beside it under its own name. E is a companion, never a
  speed.
- **Interval.** A 95 % paired bootstrap interval of `S_geo` (2000 resamples
  of the points, and of each point's samples). A mechanism is a performance
  gain only when the lower bound is above 1 and the frozen threshold is met;
  otherwise it is "no significant contribution".
- **Regressions.** A point more than 2 % slower needs a recorded cause; more
  than 5 % slower under the final default fails.
- **Refused, not flagged:** a frozen point missing or not passing on either
  side; a passing point with zero or no time; more than one configuration on
  a side; A and B differing in anything but the declared knobs or declared
  platform fields (`--vary TNCA_X`, `--vary platform.image`); a changed point
  definition (tolerance, shape, demand); changed weights; a record without
  provenance (software tree, artifacts, profile, capabilities, image, clock,
  launch map); a makespan assembled from per-core or per-kernel times; equal
  device cycles at a different clock; warm-up, autotuning or compilation
  inside a timed sample; a protocol-1 record other than the frozen B39.
- **Provenance** (`archbench/provenance.py`). Every protocol-2 record names
  its software by content (`files:` digest of the sources, the same in a git
  checkout and an rsync'd copy), every artifact by sha256, the profile and its
  capabilities, the image and clock, and the launch map. With
  `--bundle-dir`, a per-point bundle holds the Dynamo graph, the Inductor
  wrapper with its Triton source, each kernel's TTIR, TNCA IR and assembly
  (`TNCA_KERNEL_DUMP`), Triton's cache entries and the artifacts, each file
  hashed in the record; `ablate.py --audit-bundles` checks them.
- **Cost.** `ablate.py --resources A.json B.json` reports the two images'
  LUT, FF, BRAM, URAM and DSP, per chip and per SLR, and their setup and hold
  slack (`tools/r40/resources.py`) beside the speed.

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
