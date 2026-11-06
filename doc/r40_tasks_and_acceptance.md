# R40: Native TNCA Device, Scalable Multicore NUCA, and HBM

Version: 0.1, 2026-10-10. Status: **PROPOSED / NOT ACCEPTED**.
Execution status (2026-10-14): R40-00 closed (`evidence/r40/R40-00/`); R40-01/04 continue to advance native device, HBM, and joint timing acceptance. Functional evidence for R40-05's global dispatcher, bounded role chunks, native three-/eight-core execution, and descriptor/pair has been submitted (`evidence/r40/R40-05/role_chunks/`). Functional evidence for R40-06's independent multi-outlets, public negotiation, and simulation traffic has been submitted (`evidence/r40/R40-06/memory_ports/`); joint functional evidence for cache line AXI and elastic links is in `evidence/r40/R40-06/line_axi/`. Acceptance items for R40-05/06 three-SLR/HBM board-level, performance, locality/DCO, etc., remain open; these functional subtasks are not equivalent to work package sign-off.
The functional checkpoint for R40-05/06 cross-region pipelining and three-way AXI board-level top is documented in `evidence/r40/R40-05/platform_pipeline/`: AXI-Lite submission, NoC random backpressure, module synthesis, and XO packaging have been verified; actual three-SLR placement/routing, on-board execution, and performance sign-off remain pending.
Starting point: end of R39; reviewed on branch `r39`, HEAD `3a546702db3fda63547b4c1ff0a9d779a6c5cfbe`, with uncommitted working-tree changes from R39. The formal baseline is frozen by R40-00; uncommitted work or un-signed images must not be treated as baseline.
Sign-off name: `R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM`.
Parent and predecessor documents: [R39 tasks & acceptance](r39_tasks_and_acceptance.md), [R39 report](r39_report.md), [ArchBench protocol](../archbench/PROTOCOL.md). Stage documents are located in `doc/`.

## 0. Phase Objective and Standard Path

R40 advances the small-scale single-core system to a complete system featuring **native device tensor residency, general Triton compilation, multiple physical cores, distributed NUCA, and real external memory**. The U280 is the initial implementation vehicle; the logical architecture and software interfaces must scale to more cores, different external memory controllers, and future ASIC implementations.

### 0.1 The Only Path for Performance Sign-off

```text
PyTorch model / op, inputs and weights as genuine torch.Tensor(device="tnca")
  → torch.compile(..., backend="inductor", fullgraph=True)
  → Inductor TNCA device scheduling, fusion, Triton templates (tensor descriptors enabled)
  → Triton TNCA backend (upstream frontend → verified TNCA lowering)
  → .tnca → tnca.runtime.Session
  → grid dispatcher → physical core / role context → NUCA / NoC → external memory controller
  → U280 HBM; deterministic RTL for functional and causal verification
```

- **R40 performance gains must originate from the PyTorch entry point above.** Handwritten Triton, `compile_aot`, IR builder, planner, assembly, and AXI traffic generators can localize issues, verify hardware, or diagnose, but their numbers cannot replace performance acceptance for any work package. Generating TNCA IR internally within shared `_lower_ttir` is normal compiler flow; replacing standard-path TTIR or injecting custom-tuned IR during measurement is forbidden.
- R39's CPU tensor → Inductor `cpu_backend="triton"` → TNCA path is retained as a migration baseline. It already generates Triton; "CPU lowering is uneconomical" refers to transfer, heuristic, wrapper, and cache limitations caused by CPU device semantics, not that computation runs on the CPU.
- Descriptor configurations must be enabled. Use cases suited to descriptors (GEMM, contiguous reads/writes) must provide evidence of actual descriptor transfer generation and execution. Indirect indexing, strides not expressible by descriptors, etc., allow legitimate pointer lowering within the same Triton kernel, recording cause and coverage; disabling descriptors across the entire graph to gain speedup is forbidden.
- Unsupported ops, dtypes, layouts, and shapes must be rejected explicitly. Acceptance graphs must have 0 CPU/ATen external fallback and 0 graph breaks; scanning wrapper strings alone is insufficient—dispatcher/runtime event audits are required. Metadata and shape computations on CPU are allowed; tensor numerical computation and implicit transfer back to CPU are forbidden.

### 0.2 The Precise Meaning of "Going via GPU Lowering"

Register `PrivateUse1` and rename it to `tnca`, establishing the allocator, device guard, stream/event, FakeTensor/Meta, and Dynamo DeviceInterface, then registering Inductor scheduling, wrapper, heuristics, and descriptor capabilities for `tnca`. Reuse upstream Triton code generation and fusion mechanisms suited for accelerators; the TNCA backend maps these tensor operations onto its own execution model.

**Device registration does not automatically select the GPU codegen path, nor does it automatically lift block width, register, and async capability limits.** The existing CPU path already uses TritonScheduling; R40 must prove that the new device identity, properties, templates, and runtime are actually used. Upstream `register_backend_for_device` is the integration mechanism; check exact signatures against locked versions. See the [PyTorch PrivateUse1 tutorial](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) and [Inductor backend registration source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/common.py).

The Triton target for `tnca` must maintain its own identity. Properties such as warp, shared memory, TMA, and FP8 may only declare semantics that are actually implemented; claiming CUDA compute capability or adopting NVIDIA layouts, PTX lowering, and hardware throughput assumptions is forbidden.

## 1. Known Starting Point and Gap to Commercial Solutions

### 1.1 Local Code and Existing Evidence

The following table is drawn from local code inspection and R39 records; on-board measurements were not rerun for this documentation work:

| Existing Capability / Gap | Evidence | R40 Treatment |
|---|---|---|
| R39 v4 standard path clearly accelerated | [R39 report](r39_report.md): tiny + small 52/52, vs. R39-00 `S_geo=4.315`; medium v4 25/25, `S_geo=4.309` in R39-13 | Use final signed-off R39 state as baseline; do not double-count past gains |
| v5 / Mode A winding down; cannot assume all new capabilities yield gains | R39-12: transpose transfers gain on tiny; Mode A slower on several short-K points; R39 report still has attention_decode board/sim 1-cycle discrepancy open | R40-00 audits, re-measures, and freezes signed-off switches; pipelining selected per workload |
| CPU device semantics and conservative configuration | [inductor.py](../python/tnca/inductor.py): CPU registered, block width 16, single 16×16×16 GEMM, persistent reduction off, graph cache off; [pyproject.toml](../pyproject.toml): Triton entry point is `cpu` | Native device integration and template capabilities accepted separately; do not just rename device |
| Storage round-trips and sync per kernel | [driver.py](../python/tnca/triton_backend/driver.py): accepts only CPU tensors, uploads all distinct storages, submit/wait, downloads; device properties contain placeholder large values | Real device storage, long-lived buffers, event dependencies, batched submission; report properties per actual profile |
| On-board execution is still on-chip RAM single-core shell | [tnca_fpga_top.v](../rtl/fpga/tnca_fpga_top.v), [tnca_fpga_mem.v](../rtl/fpga/tnca_fpga_mem.v): default 65536×8 B; [kernel.xml](../fpga/kernel.xml) has only AXI-Lite control port | Data AXI master, HBM, and bulk DMA; AXI-Lite reserved for control |
| Multicore NUCA RTL exists but lacks public software loop | [tnca_top.v](../rtl/tnca_top.v), [tnca_tile.v](../rtl/tile/tnca_tile.v): mesh, slice, TMU; terminal has only shared memory port; [architecture overview §3.2](architecture_overview.md#32-the-multi-tile-system-rtl-only) | Integrate into Session, standard path, R39 core capabilities, and multiple memory outlets; running existing top test alone does not suffice for sign-off |
| Multicore and single-core latest capabilities lack unified profile contract | Tile/top parameters differ from single-core shell R39 switches; `system-2x2` defaults without SFU | Core capability parameters thread through all layers; existing operator coverage must not regress due to core replication |
| Wide tile, allocation, and matrix throughput gaps | R39 report §6: wide block, software pipelining, bank-aware allocation incomplete; K16 MMA ~256 cycles | Close wide blocks and resource allocation first, then decide compute resource investment based on standard-path counters |
| Large scale points limited by artificial heap boundary | [ArchBench TNCA runner](../archbench/platforms/tnca.py): 384 KiB heap, cycle accumulation across launches, single first-call measurement; medium reduced shapes for this | Add HBM suite, retain original point definitions; upgrade timing protocol for concurrency and real external memory |

### 1.2 Design Insights from Commercial Architectures

The following compares mechanisms in public implementations with the **TNCA approach proposed in this document** (right column); it does not imply existing implementation or measured gains. Commercial ASIC peak, process, and power metrics cannot directly form fair head-to-head comparisons against FPGA.

| Architecture & Public Mechanism | TNCA Gap | TNCA Approach in This Phase |
|---|---|---|
| Meta MTIA-2i: Dedicated Triton backend, Inductor generation adjustments, async fixed-function units, software pipelining, PID-to-PE mapping; paper includes compiled model evaluation ([MTIA paper §3–5, §7](https://arxiv.org/html/2608.00325v1)) | TNCA lacks native device system, general resource model, and physical multicore grid mapping | Native device + common compilation capability contract + automatic pipelining and locality mapping; performance measured solely via Torch generation path |
| AWS Trainium2: Multiple NeuronCores, HBM, and DMA; NKI SBUF/PSUM hold operands and accumulator results, compiler manages resources and spill ([Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html), [NKI programming model](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/programming_model.html)) | Window / accumulator / scratch lifetimes conservative; expanding blocks easily exhausts resources; memory and compute serialized | Explicit resource lifetime, staged buffering, controlled spill/retile, and async transfer; retain TNCA's NUCA design |
| Google TPU: Matrix, vector, scalar division; HBM feeds matrix, internal data reuse ([TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)) | GEMM unit throughput and reuse deficit becomes more evident after increasing core issue rate | Improve block reuse, layout, and fusion before adding resource-audited matrix throughput |
| NVIDIA Hopper: Descriptor-driven TMA, async transaction barrier supports transfer/compute overlap ([Hopper architecture overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)) | TNCA has descriptor/barrier primitives, but standard path underutilizes multilevel parallelism | Verify genuine async overlap, buffer ownership, epochs, and backpressure; `num_stages` field existing alone is not implementation |
| FlashAttention: Reduces external memory IO via tiling and online softmax ([Original paper](https://arxiv.org/abs/2205.14135)) | Mathematical decomposition materializes large score/probability tensors to external memory; connecting HBM still wastes bandwidth and capacity | Select auditable Triton fusion templates for standard Torch SDPA; resident KV cache, verify actual HBM byte traffic |

Accordingly, R40 prioritizes closing the **software device and memory loop, scalable execution/memory interfaces, locality, compute/transfer overlap, and application-level IO**. FP8/INT4, sparsity, training, and multi-device interconnect are deferred to later phases; existing FP32/FP16/BF16 and integer semantics must not be compromised for FPGA implementation convenience.

### 1.3 Physical Facts and Bandwidth Budget of U280

The U280 has three SLRs, with the HBM interface located in SLR0; the board provides 8 GB HBM, 32 AXI interfaces / pseudo-channels (PC), with a nominal total bandwidth of ~460 GB/s. See [U280 resource layout](https://docs.amd.com/api/khub/documents/rD6Ydbexy9OTF9vem6wnwA/content), [HBM Memory](https://docs.amd.com/r/en-US/ug1314-alveo-u280-reconfig-accel/HBM-Memory), and [U280 Introduction](https://docs.amd.com/r/en-US/ug1314-alveo-u280-reconfig-accel/Introduction).

Cross-SLR access must account for pipelining, routing, and timing overhead. AMD documentation recommends placement near corresponding HBM interfaces, avoiding diagonal crossings, and pipelining cross-SLR paths. HBM native AXI transactions are 32 B granularity; narrow writes waste bandwidth. See [Cross-SLR pipelining recommendations](https://docs.amd.com/r/2024.2-English/ug949-vivado-design-methodology/Pipelining-Considerations-for-Crossing-SLRs) and [HBM AXI considerations](https://docs.amd.com/r/en-US/pg276-axi-hbm/AXI-Considerations).

The current single-core data port at 8 B/cycle × 100 MHz is only 0.8 GB/s; three such data ports sum to only 2.4 GB/s theoretical project interface. Thus connecting HBM primarily removes the capacity bottleneck; achieving bandwidth gains requires expanding descriptor transfers, cache line paths, in-flight transactions, and memory outlets.

Each candidate delivers a bandwidth ledger: core ingress, transfer engine, slice, NoC, cross-SLR link, AXI adapter, connected PCs, stating peak read/write and sustainable measurements. The bottleneck is the minimum of serial stages; shared links and read/write contention are modeled separately; claiming the entire 460 GB/s across three cores is forbidden.

## 2. Scalable Architecture and Compatibility Contract

### 2.1 Boundary between Logical and Implementation Layers

| Layer | Responsibility | Prohibited Leaks of U280 Constraints |
|---|---|---|
| PyTorch / Triton semantics | Tensor, dtype, shape, stride, alias, descriptor, program ID, explicit sync | SLR number, HBM PC number, 512 KiB heap, requiring models to shrink to fixed small shapes |
| TNCA ISA / artifact / capability | Execution and memory semantics, true capabilities, resource requirements, version negotiation, rejectable capability requests | Hardcoding three cores or fixed channel counts into ISA semantics; hardcoding FPGA latencies into programs |
| General compiler and runtime | Symbolic shape guard, tile/role/core allocation, resource lifetime, event dependencies, logical address, legal retile/spill | Selecting lowering by benchmark name; altering numerical or alias semantics based on SLR layout |
| General system RTL | Parameterizable core/slice/controller counts, ready/valid and tags, NoC routing, home mapping, memory ordering | Requiring 2×2 or power-of-two for correctness; converging all controllers to a fixed single port |
| Board / platform adapter | XRT, HBM AXI, PC address mapping, SLR placement, register slices, CDC, clock/reset | Demanding frontend shrink dimensions, alter precision, or remove tensor descriptors |

Logical tensor descriptors carry only logical addresses and tensor layout. Physical PC allocation and routing are interpreted by the memory map / runtime / memory adapter; legal strides and views remain identical. Logical address fields retain 64 bits; PC boundary and capacity checks are explicit and testable. Address stability and physical continuity are declared separately; truncating low bits to handle HBM addresses is forbidden.

### 2.2 Core Count, Topology, and Capability Scaling

- Generic parameters include at least `NUM_CORES`, `NUM_RCTX`, `NUM_SLICES`, `NUM_MEM_PORTS`, line/flit width, MSHR/tag counts, buffer capacities, per-unit throughput, and latency. Capability is verified jointly by profile/target/runtime; actual images return fingerprint and limits.
- U280 initial official target: **3 physical cores distributed across SLR0/1/2**, each paired with a local NUCA slice; memory outlets and PC groups are determined by measured layout. 2 cores serves as an intermediate gate; 4 cores is an extension when resources permit. If 3 cores cannot achieve closure, propose an evidenced amendment; multiple role contexts cannot count as multiple physical cores.
- Home slice, program distribution, and requester tags must support non-power-of-two core counts. Designs must not rely on `core_id & (N-1)`, fixed 2-bit sources, or paired-core assumptions. Selection rules must be implementable without introducing costly general division on the address critical path.
- Functionality matrix covers at least 1/2/3/4 cores and 8-core RTL smoke; 1/2/4 external memory outlets, fast/slow/out-of-order responses, various buffer capacities. Large core count performance claims require real measurement; parameterization and elaboration success only evidence scalability capability.
- Preserve R39 supported operators, dtypes, layouts, aliases, and error semantics. Insufficient resources trigger semantically equivalent retile or attributed spill; fallback to standard TNCA compilation when optimizations do not apply. Silent CPU fallback, relaxing precision, or dropping test points is forbidden.
- Legacy ABI / profile / CPU migration paths maintain version-constrained regressions; new systems requesting capabilities missing in legacy images must explicitly reject. New capacity does not authorize legacy code to access new address ranges.

### 2.3 Memory Ordering and Completion Semantics

Using a shared NUCA with a globally unique home slice, establish verifiable producer-to-consumer visibility: kernel completion, event wait, cross-core dependencies, host copy, and necessary flush/invalidate. Completion events must wait until all observable writes reach the designated visibility domain; cannot rely on core idle alone. Hardware cache coherency is not promised for the host CPU cache, but explicit copy and wait must satisfy PyTorch visibility semantics.

Multicore atomics, store masks, cross-line read-modify-writes, dirty evictions, same-address read/write, DMA, and cache aliasing have explicit rules. Host DMA bypassing NUCA requires writeback/invalidate protocols or explicit ownership handoff. On fault, stop dispatch, drain accepted transactions, return the first attributable fault, and advance the recovery epoch; no active cores or DMA may continue writing old tensors.

## 3. Joint Acceptance and Ablation Rules

### 3.1 Four Baselines to Prevent Conflating Gains

| Baseline | Definition | Purpose |
|---|---|---|
| B39 | Signed-off R39 software, on-chip RAM image and config; CPU migration path | Migration compatibility and end-to-end comparison at original scale |
| B40-S | Native device, single core, same HBM/NUCA system and software, performance switches off | Paired baseline for software and HBM optimizations |
| B40-N | Same multicore image, varying active core mask; keeping slice/PC/cache capacity identical | Measure physical core parallelism gain alone |
| B40-F | Final candidate with all approved switches enabled; leave-one-out disabling each | Verify each improvement still contributes in final system |

Also report independent single-core to multicore product-level comparisons, recording added cores, cache, ports, and resources. Conflating fixed-resource B40-N ablation with product expansion is forbidden.

### 3.2 Every Improvement Must Have an Ablation

1. Each independent mechanism has compile/runtime/RTL switches and configuration hash. Development can bundle work packages, but **sign-off must isolate each mechanism** (wide blocks, bank allocation, pipelining, MMA widening, command replay cannot be reported only as an all-on/all-off aggregate).
2. Single-factor A/B and final system leave-one-out. Cross-layer synergies must perform legal 2×2 ablations (single/multicore × single/multi-outlet, naive/locality × DCO, pipeline off/on × single/multi-context). Record rejection reasons for illegal combinations; do not fabricate results.
3. Freeze workloads, seed, tolerance, weight, clock policy, resource constraints, and subsets. Subsets are frozen before measuring candidates based on family/shape/arithmetic intensity rules; no cherry-picking speedups post hoc. Holdout is not used to select strategy or thresholds.
4. Report point-by-point correctness, A/B time, `S_geo`, balanced weighted geometric mean, full suite E, worst degradation, counters/traffic, resources, frequency, and failures. Inherit R39's `S_geo` definition; newly added weighted metrics are named separately; E cannot substitute for speed.
5. Unless separately listed in a work package, any point regressing >2% requires per-point root cause and automatic selection/fix; points still regressing >5% in the final default policy fail. Functional changes sign off as "capability delivery", not performance speedup.
6. Performance improvements require a 95% paired bootstrap confidence interval lower bound >1; target subsets in the final leave-one-out must show ≥3% gain. Failing this is recorded as "no significant contribution" or regression.
7. Disabling new switches must preserve existing point-by-point cycle / artifact contracts on the original profile and deterministic memory path; when switching to HBM/concurrency, compare against capability, semantics, and corresponding baseline (exact cycle match is not required across different memory systems).

### 3.3 Timing, Statistics, and Counters

- **kernel makespan**: grid acceptance to all cores completed and writes in visibility domain; **graph makespan**: initial submission to final dependency completed, including device wait and scheduling gaps. In multi-stream / multicore, summing active cycles or kernel overlapping times is forbidden.
- Global timestamps use a common readable timebase; separate frequencies are recorded for multiple clock domains and converted, cannot add cycles directly. If legacy B39 cannot provide graph timestamps, compare reconstructible kernel time and host latency; do not pair legacy cycle sums with new graph makespan.
- **resident call latency**: weights and inputs resident, host call synchronized, monotonic clock measured; **request latency**: includes explicit H2D and D2H. Paired A/B must use identical boundaries.
- Cold compile / autotune, weight initialization, and steady execution are recorded separately. Board measurements: ≥5 warmups, 30 timed calls across 3 independent sessions/blocks; interleaved or randomized A/B; report median, p95, paired bootstrap CI. Long tasks: ≥10 timed across 3 blocks, sample plan pre-registered in R40-00.
- Warm resident vs cold-cache reported separately. Cold-cache flushing must cover actual NUCA capacity.
- Per-core / per-context counters: issued, active, stall attribution, busy; per-slice: hit/miss/merge, MSHR full, dirty writeback; per-link/port: flit/beat, bytes read/written, backpressure, latency, outstanding, fairness. Mutually exclusive issue/stall conservation holds per core; matrix/DMA busy overlap cannot enter the same conservation sum.
- Update ArchBench `serial-grid`, single first-call measurement, and single-port roofline assumptions. Lower bounds are calculated and versioned based on actual compute/memory/interconnect topology and semantic bytes. Measured sustainable bandwidth is used for diagnostics, not replacing peak bound. E serves as a companion metric; primary comparison uses wall-clock seconds and `S_geo`.
- Provenance saves PyTorch graph, Inductor wrapper/Triton source, TTIR, TNCA IR/assembly, artifact hash, profile/capability, software/tree hash, image/platform/clock, launch/event/core mapping, and raw samples. Every performance point must trace back through this chain.

### 3.4 Correctness, Board, and Simulation

- Merge gates inherit R39: `make r37`, core / core-vrf / core-slowmem / core-ooo / top / nuca, `r38-baseline-check`, `r38-profiler`, `r38-torch`, `r38-tdesc`, and applicable R39 role/pair regressions; governed by the actual list and counts at freeze, not stale document counts.
- Native device path passes numerics across tiny + small + medium; descriptor on/off is compatibility regression only, performance uses descriptors enabled; dtype tolerances inherit original suite; exact integers; directed tests for NaN/Inf/mask/alias.
- Original on-chip RAM B4 continues to require point-by-point cycle-exact match; R39 open discrepancies are retained, not auto-closed by switching to HBM rules.
- HBM does not promise board/sim cycle equality. Verify protocol and data correctness first; prove identical transactions and completion with fixed AXI trace replay on adapter/NoC, then calibrate memory model using board latency distribution. Pre-freeze ≥10 calibration points and ≥10 holdout points; holdout graph makespan median relative error ≤10%, error p95 ≤20%. If unmet, mark model uncalibrated; cannot use its cycles to substitute for board acceptance.
- Hardware: lint/elaboration, directed fast/slow/out-of-order/backpressure tests, OOC, then link. Final report gives per-SLR and chip-level LUT/FF/BRAM/URAM/DSP, SLR crossing, post-route setup/hold, CDC, and DRC. Core 100 MHz initial target; frequency changes require amendment, reporting iso-frequency and actual frequency results; silent downclocking forbidden.
- Board performance conclusions must come from actual U280 images; energy reported only with physical sensors and scope, otherwise null. Faults, timeouts, OOM, and unmeasured points remain in the full suite.

## 4. Work Packages and Concrete Acceptance

The following thresholds are engineering targets, **not measured results or performance projections**. R40-00 may submit evidenced amendments based on baseline measurements prior to implementing optimizations; lowering thresholds post-optimization to claim passing is forbidden. `tools/r40/*`, `spec/r40/*`, and `make r40-*` represent deliverables for this phase; this document does not assert they are currently executable.

### R40-00: Close Starting-Point Uncertainty, Freeze Reproducible Protocol

Deliverables:

- R39-12/13 final state, Mode A selection rules, and unresolved B4 evidence; freeze B39 commit/tree, all switches, image, and toolchain. Unresolved items must indicate responsibility and blocking scope; do not blend R39 v4/v5 into one baseline.
- `spec/r40/plan.json`, `baseline.json`, `suites/`, `ablation_matrix.json`: target subsets, holdouts, sample plans, formulas, thresholds, dependencies, and configuration identity. Lock torch 2.14.0 / Triton 3.8.0 and extension APIs; upstream main docs are reference only.
- `tools/r40/ablate.py` adds multiple timing kinds, manifest integrity, confidence intervals, config diff whitelists, hardware resource comparison, provenance audit; update ArchBench protocol version.
- Pre-freeze HBM suite: retain all existing points; add streaming / GEMM / transpose / gather with working sets ≥4× total NUCA capacity (≥32 MiB), plus original large tier and multi-step decoder. Record capacity, compile coverage, and measurements without shrinking original large shapes.

Acceptance:

1. `make r40-protocol`: manual removal of slow points, zero-filling unmeasured points, mixing images, changing tolerance/weights, missing provenance, adding core cycles to fake makespan, changing frequency only, injecting autotune/replay warmup must all be refused; identical inputs yield S_geo = 1.
2. B39 full numerics and existing gates reproduce; device, host, transfer, and compile time boundaries verify from logs. Frozen subsets exist before implementing optimizations.
3. `tools/r40/acceptance.py` distinguishes `capability-pass`, `perf-pass`, `no-significant-gain`, `fail`, `not-measured`; missing commands or missing evidence cannot PASS.

Progress (2026-10-10, execution record; contract unchanged):

- **R39 wrap-up**: R39 signed off, PR #6 merged into `r38-dco` (merge `b5c606d`); `r40` branches from here.
  - Final version: software `be87779` + image `u280_r39h` (v5). Relative to R39-00 baseline: 77 points S_geo 5.548; relative to v4 final: 1.287.
  - Mode A with `ring_estimate` gating: 77 points S_geo 1.0045, suite cycles −0.37%, recorded as "no significant contribution", off by default.
  - Selection rule: `rctx_roles=1` splits only when estimated profitable and roles do not concurrently read/write same memory without order; `2` forced (test only).
  - Evidence in [`doc/r39_report.md`](r39_report.md) §8.4–8.5.
- **B39 frozen** (`spec/r40/baseline.json`, `spec/r40/b39/`):
  - Software: commit `be87779`, tree, content digest `files:04338f…` (`archbench/provenance.py`, identical for git checkout and non-git copies).
  - Default switches and Inductor configuration.
  - Image: sha256, profile, capabilities, RTL parameters, clock policy, measured frequency, timing. Resources per SLR recorded in `spec/r40/b39/resources.json`.
  - Toolchain: Vitis/Vivado/XRT/torch/Triton/Icarus/Yosys versions on board host and build host.
  - Records: 77 on-board records, B1, B4 simulation references. Single configuration (v5, Mode A off), no v4 mixing.
  - 5 unresolved items, each with responsibility and blocking scope:
    - B4: llama_decode and attention_decode, both 1–2 cycles off due to cross-launch state.
    - softmax fp16 tiny sensitive to timing switches (handed to R40-07).
    - fp32 matmul/conv2d 0.7–1.4% slower on v5 (handed to R40-07).
    - GitHub Actions not triggering due to account billing.
- **Baseline correction**: §1.1 table "K16 MMA ~256 cycles" was an early R39 estimate. Measured v4 is 551 cycles, v5 `MTX_ABUF` reduced to 311 cycles ([`doc/r39_report.md`](r39_report.md) §6, §8).
- **API lock** (`spec/r40/api_lock.json`, `tools/r40/api_lock.py --check`): under torch 2.14.0, Triton 3.8.0, all 25 extension APIs required by R40-01 present, signatures recorded.
  - Includes `register_backend_for_device` (with fx wrapper and custom config parameters), `_setup_privateuseone_for_python_backend`, `register_interface_for_device`.
  - Triton discovers backend via entry point `cpu` (as noted in §1.1).
- **ArchBench protocol 2** ([`archbench/PROTOCOL.md`](../archbench/PROTOCOL.md) version 2):
  - Cold-start first call recorded separately; warmup and timed calls separated, one sample per call.
  - Timing kinds: `device-cycles-serial-grid`, `request-latency` alongside timed calls.
  - `host.steady` records steady-state upload / run / download.
  - Every record includes provenance: software digest, artifact sha256, profile and capabilities, image and clocks, launch map.
  - `--bundle-dir` stores Dynamo graph, Inductor wrapper and Triton source, TTIR / TNCA IR / assembly, Triton cache, and artifact file hashes.
  - Verified on simulator tiny add and mlp.
- **Tools** (`tools/r40/`):
  - `ablate.py`: multiple timing kinds, config diff whitelist (knobs and `platform.<field>`), bootstrap confidence intervals, bundle audit, resource comparison (`resources.py`).
  - `acceptance.py`, `protocol_tests.py`: 34/34 test cases pass. Every counterexample is rejected for the correct reason; identical inputs yield S_geo = 1, conclusion "no significant contribution"; missing commands, missing evidence, or unnamed gates cannot PASS.
  - `b39.py`, `b39_gates.py`, `b39_repro.py`, `freeze.py`, `hbm_coverage.py`, `evidence.py`.
  - Make targets: `r40-protocol`, `r40-freeze[-check]`, `r40-api-lock`, `r40-hbm-coverage`, `r40-b39-gates`, `r40-b39`, `r40-acceptance`.
- **HBM suite** (`archbench/hbm.py`, `archbench/suites/hbm.json`, `spec/r40/suites/hbm.json`):
  - 12 new points with working sets ≥32 MiB (4× the total 8 MiB NUCA architecture):
    - streaming: add / silu / copy 2^26, softmax, rmsnorm.
    - GEMM: matmul 4096³, linear, gemv on 86 MiB weights.
    - transpose: 8192² fp32.
    - gather: embedding, 8192 rows from 256 MiB table.
    - R40 decoder: 4 layers, 211 MiB weights, prefill 512; `archbench.generate --tier hbm --new 64` is the multi-step decoder workload.
  - Suite = all existing points + original large + these 12 points, totaling 113 points plus decoder workload. Original large unscaled.
  - Coverage record (`spec/r40/hbm_coverage.json`): 35 of 36 large + hbm points compile via standard path, 0 fit B39 heap, 0 measured.
  - Uncompiled point is `op.layernorm/large`: loop reduction generates `scf.if` without TNCA lowering. Handed to R40-03.
- **Frozen subsets** (`spec/r40/suites/`, 21 subsets; `plan.json`, `ablation_matrix.json`):
  - Each subset is a rule over family / tier / shape / arithmetic intensity / B39 records, resolved into a point list with definition hashes.
  - Every performance experiment has a frozen subset, variable switches, timing kinds, thresholds, and baselines.
  - R40-09 holdout is 30% of medium + large + hbm by hash (14 points); HBM model calibration/holdout has 17/19 points.
  - Remaining per-package subsets and thresholds are listed in `plan.json`.
- **Reproduction**:
  - B39 software (export of `be87779`) rerun on `u280_r39h`: 77/77 numerical pass, **77/77 point-by-point cycles identical to frozen record**.
  - R40-00 tree run under protocol 2 on same image (5 warmup, 10 timed calls):
    - 77/77 pass; first-call cycles identical to B39, zero cycle variance across 10 timed calls.
    - Time boundaries consistent: host run ≥ device time, request ≥ parts, first call ≥ steady calls.
    - Compile median 3.5 s; device time accounts for median 0.3% of request on CPU tensor path.
  - B39 against this run (via `ablate.py`, declaring only software tree difference): accepted, S_geo 1.000000, CI [1, 1], no significant contribution.
  - R39's 27 merge gates run on this tree, counts frozen as actual (`spec/r40/b39/gates.json`).
    - 3 do not exit 0 on B39 commit either, frozen with reasons: `r38-l0` (2 checks pending by design), `r38-baseline-check` (R38 baseline profile config; 149/149 test cases pass), `r38-identity` (requires `make r38-baseline` first).
- **R40-00 closed** (2026-10-10): `acceptance.py --wp R40-00` 4 gates all `capability-pass`:
  - protocol 34/34.
  - frozen 33/33.
  - b39-gates 27/27.
  - b39-reproducible: above three runs plus identity.
  - Evidence in `evidence/r40/R40-00/`.
  - Any subsequent changes to `spec/r40/` plans, subsets, or thresholds are amendments.

### R40-01: Native PyTorch TNCA Device and Inductor Accelerator Path

Deliverables:

- PrivateUse1 backend, real device storage / allocator, `empty/empty_strided`, explicit copy, view / offset / stride / alias, guard, device module, stream/event base interface; supports FakeTensor / Meta and compile-required shape guards.
- TNCA DeviceInterface, Inductor Scheduling / wrapper / device op overrides, device-specific templates and heuristics, descriptor capability, Triton `tnca` entry point and driver. Clearly isolate legacy CPU migration path, using identical TNCA lowering and Session.
- Actual profile properties replace placeholder infinite resources; multi-worker compile initialization, cache keys, device and version identity consistent; concurrent CPU/CUDA/TNCA in the same process does not overwrite configs or hijack drivers.

Ablation: A is B39 CPU migration path; B is native device with `transfer_policy=per-kernel` legitimate staging comparison; C is native residency. A/B evaluates generated code and device scheduling differences, B/C residency gain belongs to R40-02. Each comparison matches input/output boundaries; attributing total A/C gain to device registration alone is forbidden.

Acceptance:

1. `make r40-device`: `model.to("tnca")`, `torch.compile(fullgraph=True)`, output `.cpu()` runs through all tiny + small + medium; tensor device/dispatch key, device scheduler, target, and runtime buffer evidence consistent, zero numerical fallback. Run capacity-fitting smoke before HBM readiness; full sign-off completed after R40-04, without shrinking original points.
2. Mixed-device tests run on CUDA and non-CUDA hosts; import/compile order, concurrent first registration, process workers, cache hits, checkpoint `map_location`, unsupported dtype errors all reproducible.
3. Descriptor generated code and device events cover GEMM / reduction / pointwise representative points; new device/codegen on tiny + small + medium achieves device time `S_geo ≥1.0`, without unaddressed >5% regressions. This package signs off as device capability; claiming codegen speedup must satisfy §3.2.

Progress (2026-10-10, in progress; design and audit see [r40_tnca_device.md](r40_tnca_device.md)):

- **Device** (`python/tnca/torch_device/`): PrivateUse1 renamed to `tnca`.
  - C++ extension: Allocator DataPtr is a device address (buffer in runtime heap), plus empty / empty_strided / as_strided / view / `_reshape_alias` / set_ / resize_.
  - Python components: All directions, arbitrary strides and dtypes copy, fill_ / zero_ / item(); device module `torch.tnca` (stream / event are synchronous semantics for R40-01).
  - Rationale for C++: torch 2.14 pure Python backend can only build storage-less tensors, requiring view and alias to be rewritten per operator.
- **Compilation**:
  - Dynamo DeviceInterface properties taken from profile target files.
  - `register_backend_for_device("tnca", TritonScheduling, ...)` with device op overrides; kernel compiles targeting Triton target `tnca`.
  - Launcher passes `tnca` tensor's own buffer and offset directly, 0 bytes uploaded; mixed host / device tensor launches rejected.
  - Runtime code area updated to same allocator as heap, adding `Session.unload`: under resident tensors, when code area fills, unloads least-recently-used modules.
- **Lowering policy** (`tnca.torch_device.policy`):
  - `tnca` is not added to Inductor GPU_TYPES; takes CPU path (B39) answer at `is_gpu()`.
  - 6 locations checking `device.type == "cpu"` give `tnca` identical answers, with individual toggles: templates, variance, cat, small matrix decomposition (including bmm lowering), realize and fusion, conv layout. SDPA backend selection uses CPU's own selector (C++ stub).
  - Treated as GPU initially: tiny S_geo 0.834, layernorm 0.32, rope 0.31 (reasons in design document §3). Aligned after adopting above policy.
  - `scf.if` lowering (if-conversion / branching) implemented: closes compilation gap for Welford and `op.layernorm/large`.
- **Testing** (`tools/r40/device_tests.py`, 16 tests):
  - Passed in both simulation and U280 board: storage and alias, copy, module, rejection, compiled graphs (pointwise / reduction / norm / GEMM / SDPA, 0 upload), view, code area unload, properties, policy.
  - Acceptance 2: On CUDA host (GB10), CUDA and tnca graphs run alternately in same process; non-CUDA board host also tested. Three import orders, 8-thread concurrent registration, spawn workers, cache hits, checkpoint map_location (cpu / tnca), rejection of complex and fp64 calculations.
  - Plugin identity O0/O1/O2 149/149 bytes identical: changes alter no B39 artifacts.
- **Device time** (tiny, RTL simulation, vs. B39 CPU path): 20 of 27 points cycle-identical, S_geo 0.999. Differences within 2%, each explained:
  - softmax fp16, gpt2_block, attention_prefill: Inductor inserts `tl.debug_barrier()` before read-after-write in reduction loop for non-CPU devices; TNCA compiles to fence (~20 cycles each). This is the synchronization B39 lacked (B39-OI-6).
  - resnet_block −1.7%: FakeTensor conv backend gives contiguous output for PrivateUse1, while CPU (MKLDNN) gives channels-last.
- **On-board** (`u280_r39h`, `TNCA_TORCH_DEVICE=native`, protocol 2):
  - tiny 27/27, small 21/25, medium 20/25 pass. Remaining 9 points capacity-exceeded: resident tensors do not fit on-chip 384 KiB heap.
  - 68 points vs. B39: S_geo 1.0015, 52 cycle-identical, worst 0.983.
  - Request latency on tiny is 3.59× faster than B39 CPU path (geometric mean, range 1.66–17×).
- **Gates** (`evidence/r40/R40-01/`):
  - `R40-01.mixed-device` capability-pass: 16/16 on CUDA and non-CUDA hosts each.
  - `R40-01.device` and `R40-01.device-time` are not-measured (9 points capacity-exceeded), completed after R40-04 per acceptance 1.
- **Descriptors on graph inputs** (`TNCA_INPUTS_ALIGNED=1`, measured): on medium, rmsnorm 0.62, softmax 0.92, transpose 0.83. Caused by descriptor loads for one program per row being costlier than vector loads; disabled by default, deferred to R40-03 multi-row blocks.

### R40-02: Tensor Residency, Bulk DMA, Events, and Batched Submission

Deliverables:

- Storage lifetime spans entire graphs and multiple invocations; device allocator is reusable and stream-safe. Transfers occur only at explicit input, output, and checkpoint boundaries; weights, intermediates, and KV buffers do not round-trip per kernel.
- XRT bulk data movement and generic copy interface; AXI-Lite sends only control. Dependencies linked by events, safely deferring buffer reclamation; batched command queue, supporting guard-protected graph replay, invalidating correctly on epoch / shape / alias mutations.
- Copy/compute overlap is an independent toggle; switching synchronous queues to asynchronous and measuring only submission time is forbidden. Cross-stream usage, in-place updates, and re-copy after CPU writes explicitly record ownership.

Ablation: On the same native device graph, compare `per-kernel / resident`, `single-submit / batched / replay`, `copy-overlap=0/1` independently; host control and DMA improvements measured separately. Report both initialization and steady-state.

Acceptance:

1. `make r40-residency`: intermediate tensor H2D/D2H bytes are **0** for chained graphs and multi-step decode; steady-state re-transfer after weight initialization is **0**; output downloads restricted to explicit requests.
2. On pre-frozen ≥3-kernel graph subset, resident vs. staging steady request latency `S_geo ≥1.5`; batched/replay on independently frozen small kernel graph subset resident call latency `S_geo ≥1.2`. Full suite degradation handled per §3.2.
3. 1000 graph invocations, non-contiguous alias/offset, async free/realloc, multi-stream events, stale handles after fault recovery, replay invalidation pass; buffer high-water mark and end unreleased storage auditable, cache allocator not required to zero out.

Progress (2026-10-11, in progress):

- `TNCA_TRANSFER_POLICY=resident/per-kernel` integrated into native launcher; both use identical standard path artifacts. Staging allocates temporary device buffers for each distinct storage, transferring host bytes back and forth, sharing temporary buffers for offset/view aliases; copies back to original storage upon completion and records all actual transferred bytes.
- `tools/r40/residency_tests.py`: resident zero-kernel H2D/D2H, staging actual round-trip, artifact identity across policies, non-contiguous offset/alias in-place updates across policies, and unknown policy rejection pass 6/6.
- `TNCA_SUBMIT=batched` uses bounded runtime queue and real completion events; intermediate storage reclaimed after final task releases lease. Stream context conveys actual logical ID; single-core currently guarantees order through shared FIFO, claiming no physical copy/compute overlap.
- Allocation tokens separated from physical addresses; reallocating same address after recovery prevents stale tensor/deleter hitting new storage. Async lifetime 2/2, stale handle isolation across three recovery epochs 3/3, payload concurrency and mutation 2/2; native device full regression 16/16, runtime lifetime and batch regressions 7/7 each.
- `make r40-residency` is currently a capability checkpoint; frozen chained graphs, multi-step decode, replay full stress, high-water mark ledgers, and performance gates are not yet complete; R40-02 cannot be signed off on this basis.
- Native batched mode 1000 invocations passed: bounded queue, numerics, end storage count, and lease audit passed; stream context/events and zero-kernel transfers passed 4/4 in group. Concurrent thread audit passed 5/5 in separate group.
- Standard Torch launch captures and preserves intermediate storage via `torch.tnca.graph(Graph())`; replay reuses identical module/parameters/artifacts, bypassing wrappers, while logs still wait for real completion. Shape/stride/storage alias and recovery epoch validated whole-graph before first admission; failure requires reset/re-capture. Verified 4/4, including updating inputs and partial execution rejection negative tests.
- Allocator retries after earliest task releases retired storage when capacity is insufficient with active tasks pending; capacity checks and allocation reside in same Session lock. Under 393,216 B real on-chip heap, 12 invocations passed 2/2: all 12 capacity backpressures remained below 8-task queue limit, actual retained bytes peaked at 328,704 B, ending at 65,536 B. `Session.memory_stats` includes retired leases and scratch, reporting heap/code high-water marks per epoch. Allocator recovery 3/3, lifetime 2/2 re-verified; concurrent 1000-call 5/5.
- 2026-10-12 checkpoint: Retained graph real replay 1000 times passed, verifying artifact identity, zero kernel transfer, heap/code high-water mark, and trailing leases. Frozen tiny attention_decode and llama_decode steady state across both policies passed 4/4 each; actual Session transfers match launcher ledger (zero for resident, real round-trips for staging). `make r40-frozen-residency` adds isolated process entry for complete 24 points, retaining per-point logs; initial full run 10/24, remaining 14 timed out at 600 s host wait, retaining failure logs; full rerun with extended host wait running on jump host, device cycle deadline unchanged.
- Continuous KV growth checkpoint: Standard fullgraph Torch 3 steps passed, KV 16→19; 45 actual kernels per step, steady Session tensor transfer zero, cache addresses and weight storages unchanged. 64-step run in progress; physical copy overlap, online attention, and performance gates remain open.

### R40-03: General Tiles, Layout, and Resource Allocation

Deliverables:

- 1-D blocks of at least 16/32/64/128 elements spanning multiple vector registers; multi-row reduction; at least two logical GEMM tiles larger than 16×16 formed from existing MMA fragments, with K16/K32/K64 selected per verified capability.
- Bank-aware allocation, window / descriptor / barrier / accumulator / scratch lifetime management; excessive pressure triggers attributable retile / spill, distinguishing memory hierarchy level and actual bytes; switching to IR builder GEMMs ad hoc is forbidden.
- Shape/stride/tail/layout transformations handled by generic rules; GPU-generated layouts unsuitable for TNCA converted in shared compilation pipeline. Dynamic shapes handled via guards/recompilation; permanent small-shape restrictions are forbidden.

Ablation: `max_block`, multi-row reduction, GEMM tile, bank allocation, lifetime reuse, spill strategy evaluated individually, ensuring CPU/GPU migration gains are decoupled from algorithmic changes.

Acceptance:

1. `make r40-tiles`: cross-register tails (17/31/33/65/127, etc.), transpose/stride, masks, int64 index, aliases, and existing dtypes correct; insufficient pressure triggers legitimate retile/spill, resource accounting matches machine demands.
2. Pointwise/reduction frozen medium + HBM subset vs. narrow blocks `S_geo ≥1.3`; larger GEMM tile frozen reuse subset `S_geo ≥1.15`. Individual mechanism final contributions verified per §3.2.
3. When switches are off, legacy profile artifact contracts pass; new capabilities must not depend on U280 PC / SLR or benchmark family names.

Progress (2026-10-11, capability checkpoint, not signed off):

- Standard Torch path 1-D block tiling across VLEN registers; 16/32/64/128 pass 12/12 each, including tails, int64, fp16/bf16/i8/i16/i32, stride/offset/alias, and in-place updates. Tests verify actual artifact widths to prevent treating capability declarations as execution widths.
- Predicates reconstructed per actual capacity; long-lived vectors in straight-line IR can spill to private per-program scratch. 128-wide int64 graph triggers 896 B / 14 slots, artifacts show 52 VRF per role across two roles, with correct numerics. Control-flow spill remains rejected; fixed issue where RCTX fallback dropped plain kernel final retry.
- Wide single-row reduction sum/amax/RMSNorm/softmax/LayerNorm pass 20/20 each at 32/64/128; generic multi-output argmax fold passes 4/4 each at 16/32/64/128, covering NaN and first equal maximum index. Scalar-carry loop preserves iteration memory ordering; actual width/kind recorded in artifact provenance.
- Independent `TNCA_REDUCTION_ROWS` cap defaults to 1; two-row fragments of the five reductions pass 20/20 each at lane caps 16/32/64, including tails and zero resident launch transfers. Two-row argmax passes 4/4 each at 16/32; double-word calculation/writeback with index offset `2^40` passes another 4/4. Fixed tile select mistakenly sending scalar predicates to `v.splat`.
- Default core-sim O0/O1/O2 artifacts match independently reconstructed frozen B39 references byte-for-byte (447/447); RCTX SPMD/roles pass in both v4 and v5. Reference commits and full inventory checked by `tools/r40/contract_tests.py`; this does not constitute a performance measurement.
- Standard native Torch mm/addmm in logical 16×16, 16×32, 16×64 at K16 pass 4/4 each; static shape guards cover 17×65×33 and 33×31×65, resident launch transfers are zero. Artifacts confirm actual wide tiles; shared A fragments reused only within same logical dot to avoid repeated packing consuming windows. Descriptor tiling offsets only trailing coordinate, passing genuine Triton→RTL regressions where both coordinates share zero; default artifact contract re-verified 447/447.
- RTL harness directly verifies post-kernel memory; mutations restoring coordinate errors are caught. Compiler RTL passes 68/68, covering legal 26 B strides and illegal rank error propagation; issued counters use identical running ROI after async faults, preserving cycle partition precision.
- Bank allocation checkpoint: `TNCA_BANK_ALLOC=1` prioritizes low-conflict temporary registers per profile's real 4-bank contract; search is bounded, preserving window, role slice, and ownership checks. Defaults to off; when off, B39 contract is 447/447; when on, 64/128 pass 12/12 each. Native Torch triad / LayerNorm / f32 dot pass 3/3 in both modes; final assembly static conflict readings are 8→0, 19→1, 110→110 respectively, with no added moves. f32 dot shows no cycle gain, register prefix 41→42; this is not a frozen performance sign-off.
- 2026-10-12: FP16 16×16×32 and 16×32×32 standard mm/addmm pass 4/4 each, row/column chunks repack loaded values per generic rules; actual K32 MMA encoding verified, default B39 artifacts remain 447/447. BF16 passes 4/4 each across both tiles; N32 retains successful re-measurement after initial host 600 s timeout, device deadline unchanged.
- Single IR role `TNCA_WINDOW_REUSE=1` reuses physical registers for awaited, non-overlapping window lifetimes, including alias reads, loop back edges, and async reader/writer completions. Unwaited, conditionally awaited, and cross-role sharing are explicitly rejected, keeping general temporary registers separated from windows. 11 safety/ledger test cases pass; 16×64×32 standard mm/addmm passes 4/4, actual machine code uses K32, logical windows reduce from 80 to 32 retained. Machine code audits across six retained artifacts confirm four GEMMs each use 47 vector registers; new opt-in artifact ledger accounts for this union; default contract remains 447/447.
- K64 resource retile checkpoint: When windows are open and full operands occupy VRF, selects the maximum fragment among target-supported K steps that retains private registers. 16×16×64 FP16 mm/addmm passes 4/4; executes two K32 MMAs in K order, sharing the same accumulator, using existing immutable row/column chunks without re-reading inputs or replacing TTIR. Actual window reduces from 64 to 32, VRF 44/45; default contract remains 447/447. Packed input windows remain rejected, BF16 re-verification running.
- Descriptor/barrier lifetime, broader retile/spill, and frozen performance gates remain pending. Evidence in `evidence/r40/R40-03/`, default block cap remains 16.

### R40-04: Single-Core Real HBM and Scalable External Memory Interface

Deliverables:

- Generic line/block request-response interface with source/transaction tags, byte masks, backpressure, error, and completion semantics; board-level HBM AXI adapter provides burst, multiple outstanding transactions, 4 KiB/PC boundary splitting, independent read/write tracking, and CDC.
- Descriptor block movement targets at least **32 B/core cycle** ingress capability; scalar LSU retains 8 B and mask support. Line/beat width is parameterizable; merging and splitting preserve observable semantics. Burst, outstanding depth, and wide ingress evaluated via separate ablations.
- All 32 PC address maps and capacities have auditable coverage; active PC count selected per budget, avoiding unreachable regions; heap resides in real device memory, with protected boundaries for code/descriptor/control.
- XRT BO / bank placement corresponds to device addresses; high address bits, memory aliases, and host DMA visibility tests cover PC boundaries. Generic interfaces provide deterministic memory models and alternative configuration adapter/models, keeping HBM protocol specifics out of core logic.

Ablation: Capacity expansion is recorded as capability delivery. On the same resident HBM system, perform standard path A/B comparisons for `burst=0/1`, `outstanding=1/N`, and narrow vs. wide descriptor ingress. Comparing on-chip RAM vs. HBM latency is treated as migration overhead; claiming speedup based on capacity expansion alone is forbidden.

Acceptance:

1. `make r40-hbm`: standard Torch graph weights or working set ≥64 MiB, substantially exceeding on-chip RAM, with non-zero actual HBM traffic; full address space host/device sampling and high-bit address tests pass; testing only the first PC is insufficient.
2. Standard Torch streaming read, write, and copy exceeding cache size by 4×: effective bandwidth in each direction ≥**50%** of frozen bottleneck peak; independent wide ingress / concurrency improvement target subset `S_geo ≥1.5`, reporting burst efficiency and read/write traffic amplification.
3. Out-of-order responses, random backpressure, subword stores to identical lines, errors, timeout/reset, cross-boundary transactions pass; HBM model fidelity satisfies §3.4, on-board data is correct, and timing/CDC gates pass.

Progress (2026-10-10, in progress; design, verification, and bandwidth data see [r40_hbm.md](r40_hbm.md)):

- **Interface**: Core data port remains unchanged, with a bridge connected behind it.
  - Addresses below `LOCAL_LIMIT` (0x20000) route to on-chip RAM, including code and parameters.
  - Remaining addresses route through AXI window: `axi_base + (A − LOCAL_LIMIT)`, where host writes `R_AXI_BASE`/`R_AXI_SIZE`.
  - New registers 0xA8–0xD8: window, sticky error/unowned response, AXI read/write/read-latency/R-beat counters. Legacy images read 0.
- **Two bridges** (`burst` mechanism A/B, `AXI_LINE`):
  - `tnca_axi_bridge`: single-beat, word-level hazard table; atomics perform read-modify-write inside the bridge; 1024-cycle quiescent window after reset drains in-flight legacy responses.
  - `tnca_axi_bridge_lb`: reads in 64 B lines, INCR burst into fully associative line buffer with next-line prefetch. Writes are direct and FIFO-buffered. `inv` pulses one cycle on each command to ensure visibility after host DMA.
  - Ordering rules and mutations: 5 mutations for single-beat bridge and 6 for line-buffered bridge, all caught.
- **Simulation**: Memory mode `core-vrf-sync-v5/hbm`. Compilation, capabilities, and artifacts follow base profile; bench includes 32 MiB AXI window and adjustable timing AXI model.
  - `make r40-hbm`: bridge passes 7 test cases across 4 geometries, covering out-of-order, backpressure, SLVERR, in-flight reset, and post-DMA invalidation.
  - Device tests pass 16/16, and 5/5 under hostile timing; graphs with tensors at 24 MiB of the window execute correctly; 32 samples across entire window write and read back correctly.
  - `DATA_AXI=0` unchanged: B4 sim tiny 27/27 cycle-identical to B39; R39 merge gates rerun against frozen counts; `r36-fpga-tests` expanded for both `DATA_AXI=0/1`.
  - Fixed pre-existing failure in `R36-FPGA-08` on HEAD: hardcoded out-of-bounds counter index 20 became an actual counter after R39-00 counter additions.
- **Board**: `rtl/board/tnca_hbm_xo.v` and `fpga/kernel_hbm.xml`; device runs directly on ap_clk.
  - Link configuration: `sp=tnca_u50_1.hbm:HBM[0:31]`.
  - Runtime allocates window BOs in connected banks based on xclbin group connectivity, writing base addresses; transfers in window use XRT DMA, heap bounded by device-reported size.
  - First image `u280_r40a` (single-beat bridge, v5 parameters): OOC 100 MHz WNS +1.367 ns; link failed on 2026-10-11 (WNS −0.683 ns), producing no loadable xclbin. OOC primary clock mistakenly leaked into full implementation; repair and re-measurement documented in [`doc/r40_hbm.md`](r40_hbm.md) §7. Board is shared, check occupancy before loading; author authorized on 2026-10-11 to continue advancing to R40 acceptance and commit/push.
- **Bandwidth** (simulation, AXI latency 30 cycles):
  - Line buffer with 16 lines / 32 waiters / OUT_N 16: read/write/copy achieve 91% / 81% / 77% of on-chip RAM; at 1-cycle latency, cycle-identical to on-chip RAM.
  - Ceiling is the core itself: 16-element blocks achieve only 2.3–3.7 B/cycle (~30–45% of 8 B ingress). Thus acceptance 2's 50% threshold depends on R40-03 wide blocks and 32 B ingress; modifying the bridge alone cannot reach it.
- **Three-core feasibility** (§5 requires early verification): single core occupies ~64.6% LUT of SLR1 (279k).
  - Three-CU link (one per SLR, each connecting HBM) started, but concurrent link with single-core left only ~10 GB shared host memory (competing with other users' Vivado); halted, pending single-core completion.
- **Gates**: Simulation components of `R40-04.robustness` and `R40-04.hbm` pass, both evaluated as `not-measured` awaiting board execution; `R40-04.bandwidth` is `not-measured`.

### R40-05: Multi-SLR Physical Multicore in Common Runtime

Progress (2026-10-15, not signed off): Synthesis of candidate `09ebe06` with retained queues completed; actual DCP full hierarchy/SLR assignment contract check passed, retaining all three sets of private code command/reply queues. The 60 RTL files of the new XO match that commit byte-for-byte; started 100 MHz `SSI_SpreadSLLs` link. Previous candidate failed due to global congestion level 7; raw logs retained. DCP assignment check is not actual placement/routing; passing this check does not imply physical or performance gates have passed.

Deliverables:

- Unified single-core / multicore capability and core parameters threading through, preserving R39 signed-off SFU, descriptor, role-SPMD, pair / slice capabilities; private code, parameters, scratch, and resource isolation per core.
- Global grid dispatcher: one logical launch distributes global program IDs, each program executing exactly once; intra-core roles remain an independent layer. Finite queues allow continuous task fetching, removing legacy 4096-program hardcoded limits or reporting per true streaming capacity.
- 1→2→3 cores on board, cross-SLR pipelined / elastic links; generic topology and routing independent of SLRs, with board layer defining placement. Requester tags / homes / dispatch correct for three and eight cores.
- Global completion, multicore performance counters, launch ID / event / fault / drain protocols; host serial MMIO launch per program is forbidden.

Ablation: On the same three-core image, compare `active_cores=1/2/3` with fixed role counts. Separately compare keeping total PC/NUCA constant vs. scaling with cores. Static vs. dynamic grid distribution evaluated as independent improvements, not bundled with core replication.

Acceptance:

1. `make r40-multicore`: standard path numerics all pass; execution evidence shows three physical cores active simultaneously, with post-route placement covering SLR0/1/2; substituting three serial calls or multiple contexts is forbidden.
2. On pre-frozen compute subset with sufficient parallelism (grid ≥ 32 × active cores), B40-N three-core graph makespan vs. one core `S_geo ≥2.1`, two cores ≥1.5; report `speedup/N`, per-core load, and frequency.
3. 1-D/2-D/3-D grids, non-divisible core counts, partial trailing batches, >4096 programs, zero-work/error inputs, mixed fast/slow programs, per-core faults and recovery pass. 1/2/3/4-core regression and 8-core standard Torch smoke pass; elaboration-only does not satisfy smoke requirements.

### R40-06: NUCA, Multiple Memory Outlets, and Automatic Locality

Deliverables:

- Eliminate fixed shared DRAM single outlets; parameterize slice / controller / PC group routing, distributed arbitration, and tag return, avoiding serialization followed by terminal fanout. Credit and virtual channel dependency proofs, random backpressure, and sustained fairness.
- Unique home, dirty ownership, atomics, and fences, DMA ownership transfer; tensor home / program mapping / address placement share generic policies. Stable layouts, cross-kernel reuse, and automatic policies must not require users to specify PCs or SLRs manually.
- DCO tensor metadata / lifetime derived from standard graphs and descriptors, connected via TMU; regular home mapping, locality, cache bypass, replacement / dead tile hints are independent switches. Tensors that cannot be proven dead must not be reclaimed early; aliases and async dependencies count toward lifetime.
- Slice capacity, MSHRs, and NoC flit/link bandwidth quantitatively tuned per §1.3 bandwidth ledger; cache and interconnect changes must report resource and clock costs.

Ablation: Single/multicore × single/multi memory outlets; naive/locality × DCO; cache on/bypass under identical total capacity and PC budgets. Separate cold streaming from warm reuse; all bandwidth measurements come from Torch graphs.

Acceptance:

1. `make r40-nuca`: cross-core producer/consumer, same-line masked stores, atomic reduction, post-eviction host copy, DMA cache updates, alias/dead hints, saturated read/write, and fault drain pass; no deadlocks, lost responses, or starvation.
2. Three-core HBM streaming subset vs. one-core iso-resource baseline graph makespan `S_geo ≥1.8`; multi-outlet vs. iso-core single-outlet `S_geo ≥1.3`. Every active port / PC group exhibits actual traffic; expanding cache to fit an entire workload to claim "HBM scaling" is forbidden.
3. Locality/DCO on pre-frozen reuse/remote-traffic subset `S_geo ≥1.1`, with at least one metric of actual remote bytes or HBM bytes reduced by ≥20%; both mechanisms ablated separately. Default policies for streaming loads must not force detours or repeatedly pollute cache.
4. Identical Torch graphs maintain semantics across 1/2/3/4/8-core functionality matrices and different memory latencies / port counts; parameter tuning requires no modifications to models, ISA opcodes, or frontend operator rules.

Progress (2026-10-14, ownership functional checkpoint, not signed off):

- Optional range maintenance in home slices writes back and invalidates intersecting cache lines, preserving dirty bytes outside partial DMA ranges. The all-home controller collects confirmations from each home and waits for the physical memory drain signal; commands/replies route via elastic links, holding new grids and coherent host requests during maintenance. Default off; runtime bulk DMA not yet enabled.
- 19/19 pass: two tag layouts, three latencies and backpressures; dirty/clean/zero/alias, high address bits, range overflow, and reset; 1/3/8-home aggregation; 160,000 differential pressure requests; 0/2-stage pipelined three-core execution, 455 dirty results, and single-byte DMA model updates verified. Evidence in `evidence/r40/R40-06/cache_maintenance/`.
- MMIO opcode 18, capability, shell/AXI drain/error, and common runtime ownership before BO read/write connected. Legacy images reject DMA; new path is opt-in. Real three-core platform RTL BO bypass model, partial dirty neighbors, and standard Torch 4097-element / 257-program graph pass: peak active = 3, actual traffic across three outlets, steady-state tensor staging = 0. Evidence in `evidence/r40/R40-02/multicore_dma/`. XRT board testing, independent copy overlap, timeout/recovery, and frozen performance remain pending; the BO model is not an XRT time measurement, and legacy images do not automatically gain new capabilities.

### R40-07: Asynchronous Pipelines, Layout Moves, and Matrix Throughput

Deliverables:

- Use real standard-path profiles to distinguish DMA wait, MMA service time, issue slots, VRF, accumulator, and NoC/HBM backpressure. Retain R39 negative findings: Mode A does not automatically enable when short-K and Mode B already hide latency.
- Generic loop eligibility, 2/3-stage buffers, producer/consumer or single-role pipelining; barrier/epoch/buffer ownership verification, with compiler reporting overlap plans and rejection causes.
- Descriptor transpose, register pack / layout conversion, and epilogue fusion, reducing reshuffling through global scratch; MMA compute/load overlap or widening selected per resource budget, preserving existing precision.
- Each implementation updates unified latency/throughput/resource models. If matrix modifications yield no effective standard-path gains, retain explicit negative results and resource analyses; passing work packages based on microbenchmark improvements alone is forbidden.

Ablation: Stage count, role pipelining, transpose/pack, epilogue fusion, MMA modification evaluated individually; `pipeline × context_count` and `pipeline × MMA` interaction comparisons traceable in both generated code and hardware.

Acceptance:

1. `make r40-pipeline`: long K / tail / masked tiles, different stage counts, fast/slow/out-of-order HBM, barrier epoch wrap, fault drain pass; device traces show temporal overlap of DMA and MMA rather than async opcodes alone.
2. Long-K GEMM standard path subset pipelining vs. serial `S_geo ≥1.2`; selected matrix modifications vs. iso-tile / iso-software / iso-frequency baseline compute-bound subset `S_geo ≥1.2`. Resource overheads, actual second-level gains, and final LOO ≥3% satisfied simultaneously.
3. Short-K and low-parallelism points selected by frozen automatic policies; full tiny + small + medium exhibits no unaddressed >5% regressions. If 100 MHz target does not close, gate per actual frequency; reporting cycles alone is forbidden.

Progress (2026-10-14, functional checkpoint, not signed off):

- Common AOT/JIT lowering integrates `rctx_stages=2/3`; explicit options, environment, and cache keys are consistent; parameter checks 18/18, default artifact contract preserved at 447/447. Legitimate rejection diagnostics for 3-stage ring on odd-K, cross-program epoch, single-step, and carried-pointer pass 4/4.
- Standard native Torch fullgraph mm/addmm, under 16×16×48 and 17×31×65 shapes, pass 4/4 numerical, actual ring/role, and steady-state zero staging for both 2-stage and 3-stage. Selecting measurement intervals by runtime launch sequence, actual engine traces show DMA/MMA overlap of 1272 and 1664 cycles, respectively; raw traces and actual artifacts can be re-verified. Overlap cycles do not equal speedup and cannot substitute for frozen ≥1.2 performance, LOO, automatic policy, or hostile HBM/fault coverage. Evidence in `evidence/r40/R40-07/`.
- `make r40-pipeline-checks` executes functional diagnostics; formal `make r40-pipeline` requires full frozen gate evidence. Unmeasured on-board performance remains not passed.

### R40-08: Standard Torch Attention, Reductions, and Resident KV

Progress (2026-10-14, validation entry point, not signed off): `make r40-attention-checks` organizes the frozen 15 legacy attention/llama/GPT-2 graphs, all existing online SDPA checks, and 64-step bucket KV checks. Legacy graph results additionally verify point hash, native device, descriptor, and zero fallback/staging; logs are retained independently, and single functional samples do not serve as performance evidence. `make r40-attention` continues judging all R40-08 gates; lacking board testing, complete shape/traffic, and prefill performance evidence, it remains not passed.

Progress (2026-10-15, not signed off): Online template extended to 16/32/64-dimension FP16/BF16 heads, sequentially consuming 16-column K/V fragments. `TNCA_SDPA=fused/decomposed` standard configuration supports repeated switching; wide-dim native decode/mask/GQA/MQA functional batches pass 6/6; config switching and auto-enabled D32 cases pass, with remaining auto-enabled cases running. This batch retains evidence of actual descriptor loads, registers, allocations, and zero staging; it does not serve as acceptance for frozen long-sequence performance or full models.

Deliverables:

- `torch.nn.functional.scaled_dot_product_attention` lowered through device / Inductor templates generates fused Triton, with descriptor tiled loading, online softmax, and stable reductions; insufficient capability falls back to standard mathematical decomposition with explicit logging.
- Eliminates materializing full score / probability matrices to HBM; GQA/MQA, causal, tail, and masks verified per frozen scope. R40 inference scope specifies dropout=0; unsupported cases rejected explicitly, while previously supported cases must not be removed.
- KV cache genuinely resident, supporting multi-step writes/reads and event ordering, with position supported by tensor/guard; stable shape buckets / capacity policies restrict recompilation; uploading full KV caches per token is forbidden.
- Reduction/SFU fusion and online attention ablated separately; speedups resulting from relaxing softmax precision must not be counted as architectural improvements.

Ablation: On the same standard Torch SDPA, compare decomposed vs. fused; KV copy vs. resident; fused vs. unfused reduction. Compare under identical precision, inputs, masks, and outputs, reporting intermediate HBM bytes and peak device allocation.

Acceptance:

1. `make r40-attention`: legacy attention / llama / GPT-2 graphs all pass; new long sequence, odd lengths, fully masked rows, GQA/MQA, 64-step cache updates match reference; CPU numerical fallback is 0.
2. Frozen long-sequence prefill subset vs. standard decomposition `S_geo ≥1.5`; allocation and traffic records prove no full `L×L` score/probability tensor materialization, with measured intermediate HBM bytes reduced by at least 50%. Total allocator high-water mark alone is insufficient to prove absence of materialization.
3. Decode steady-state does not retransmit weights/KV; timing covers all layers and dependencies, with correct cache updates. Recompilations across positions within a fixed bucket equal 1; guard boundaries requiring recompilation reported separately.

### R40-09: Budgeted Automatic Selection, Applications, and Scaling Validation

Progress (2026-10-14, foundational implementation, not signed off): Common candidate selection supports capability/resource pre-filtering, shared-thread candidate budget, 600-second graph budget, and full configuration cache key. Only common device clock endpoints from the same session/epoch are accepted; candidate measurement and formal launch logs are isolated. Multicore grid backend endpoints are passed to runtime Events; missing timing is rejected. `make r40-autotune-checks` serves as the functional validation entry point; auto-compiled candidates, cost model, worker enforced timeout, frozen holdout/full suite/model, and board performance remain pending; formal `make r40-autotune` judges all R40-09 gates after functional checks; lacking the above evidence, it remains not passed and basic checks cannot substitute for full acceptance.

Progress (2026-10-15, not signed off): Persistent candidate cache supports atomic publication, complete identity invalidation, and corruption recovery; native three-core standard graphs prove that reopened caches do not re-measure and formal launches have zero staging. Shared worker graph budget preserves the earliest deadline; SQLite transactions ensure multiple processes combine for at most 16 admissions per kernel; policy checks pass 14/14. This budget operates as cooperative admission; terminating timed-out workers is not yet implemented.

Deliverables:

- Valid candidates for template/tile/stage/core count/placement filtered by capability and cost models, permitting budget-constrained autotuning timed via on-board device events; sorting by RTL simulator wall time is forbidden.
- Compare conservative fixed configurations, model predictions, and measured autotuning; cache keys include shape/stride/dtype, compile/runtime/RTL capabilities, memory map, clock, and version. Candidate measurement logs isolated from production runs, with budget overheads listed separately.
- Full results for original tiny/small/medium, full original large, and pre-frozen HBM suite; at least one ≥4-layer decoder inference with weights ≥64 MiB, prefill ≥512, continuous decode ≥64 steps, using real Torch models and standard compiled graphs. Model size, weight provenance, initialization seed, numerical references, and logits validation frozen in R40-00.
- ASIC scaling design notes: bandwidth/capacity/NoC budgets for 8/16/32 cores under identical logical interfaces, projected bottlenecks, and new controller requirements; 8-core functional demonstration, with remainder marked as analytical. FPGA LUT/DSP or OOC timing cannot be described as ASIC PPA.

Ablation: Fixed config, model selection, and measured autotuning evaluated on same holdout; combined with B40-F leave-one-out into final mechanism table. End-to-end applications provide comparisons across single/multicore, residency, locality, pipelining, attention, and replay.

Acceptance:

1. `make r40-autotune`: candidates exceeding resources rejected before execution, selection independent of benchmark names; cache hits, config invalidation, worker behavior correct. At most 16 measured candidates per kernel, ≤10 minutes tuning per graph; budget overruns explicitly recorded; pre-registered long-graph budget exceptions cannot be filled post hoc.
2. Frozen holdout vs. conservative config `S_geo ≥1.1`, selection regret p90 ≤15% (relative to actual optimum among valid candidates within same budget); compilation + tuning costs and amortizing invocation counts fully reported.
3. Tiny + small + medium all correct; HBM core suite fully correct after R40-00 freeze with on-board measurements covering streaming, GEMM, transpose, gather, norm, SDPA, and decoder. Original large retains all points; unpassed large points prevent full suite certification.
4. ≥4-layer model prefill, decode, logits, and explicit boundary copies all correct; report TTFT, decode median/p95 ms/token, tokens/s, peak HBM, input/output/weight/KV traffic. Same final software and image on three cores vs. one core model graph makespan `S_geo ≥1.5`, decode regressing ≤5%; failures cannot be signed off using kernel-only speedups.

### R40-10: Board Final Acceptance, Full Ablation, and Reproduction Package

Deliverables: [`doc/r40_report.md`](r40_report.md), complete gate JSON, baseline amendments, standard path provenance, raw samples, per-mechanism A/B / LOO / interaction tables, negative results, resources/timing, HBM model calibration, and run instructions. Large generated artifacts placed in `sim/build/r40/`; Git saves spec and reproducible summaries; long-term board evidence follows `evidence/` convention and verification checklist.

Acceptance:

1. `make r40-acceptance` executes capability/correctness/performance gates across work packages; all required gates must pass for sign-off. Passing device registration, HBM attachment, RTL multicore, or compile-only independently does not constitute R40 completion.
2. Three-core U280 final image passes conformance, full original suite, and HBM core suite; every performance claim reconstructible from Torch sources and samples; every independent improvement has qualified ablations. Non-contributing mechanisms explicitly removed or retained as disabled experimental configs.
3. Main performance table uses wall-clock seconds: iso-final-software three-core vs. one-core, B39 vs. final system common point migration, and per-mechanism LOO; capacity expansion, resource additions, and codegen improvements listed separately; adding speedups together or using legacy E ratios to represent new system speed is forbidden.
4. Compare against available GPUs on identical Torch points/precisions, using strong Inductor/cudagraph and Triton-only baselines under identical request/resident/makespan boundaries; state differing resource and power envelopes. Public commercial ASIC figures used only for mechanism comparison; speed ratios not asserted without empirical measurement on identical tasks.
5. Clean-environment reproduction and board execution verified by at least one independent session; missing boards or measurements mark corresponding gates as `not-measured`; estimating throughput via RTL is forbidden.

## 5. Execution Order and Definition of Done

```text
R40-00 freeze protocol and R39 baseline
  ├─ R40-01 native device → R40-02 residency/events/submission → R40-03 general tiles/resources
  └─ R40-04 single-core HBM/scalable interface → R40-05 multi-SLR multicore → R40-06 NUCA/multi-port
                         two lines merge → R40-07 pipeline/matrix → R40-08 attention/KV
                                       → R40-09 autotune/applications → R40-10 final sign-off
```

Interfaces and correctness can proceed in parallel; optimizations lacking acceptance must not claim performance passes prematurely. Full residency and bulk DMA acceptance in R40-01/02 depend on R40-04 external memory capacity and data interfaces; standard-path performance in R40-04 also depends on R40-01. Close functional loops for both using small graphs and conservative configurations before tackling full performance gates. Verify single-core HBM and three-core resource/timing feasibility early to avoid discovering SLR or bandwidth budget shortfalls after software completion.

Each milestone updates this document's status, gate checklist, and evidence registry:

- **M0:** Starting point, capability, and timing contracts reproducible; legacy paths correct.
- **M1:** Native device + resident tensors pass on standard path; single-core HBM large tensors runnable.
- **M2:** Three physical cores across three SLRs, common runtime, NUCA, and multiple memory outlets pass; functional evidence for 1/2/3/4/8 cores complete.
- **M3:** Wide tiles, pipeline/matrix, attention, auto-selection, and model gains independently ablated; negative results not hidden.
- **M4:** Final on-board full acceptance, reproduction package, and report completed; `R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM` can then be signed off.

R40 completion requires satisfying native device, compatibility, scalable interfaces, multi-SLR physical multicore, real HBM, standard-path applications, and point-by-point ablations simultaneously. Reducing core counts, removing features, shrinking workloads, modifying tolerances, or eliminating performance gates requires public amendments; incomplete items remain marked incomplete and cannot be closed by redefining "standard path" or "multicore".

## 6. Specific Failure Modes to Avoid in the First Round

| Failure Signal | Mandatory Action |
|---|---|
| Switching to `tnca` but still using CPU heuristics / per-kernel IO | Revert to device scheduling and storage evidence; R40-01/02 cannot sign off |
| Three cores active but all serialized waiting on single memory arbiter | Isolate via single/multi-outlet ablation, expand slice/port/NoC; total active cycles cannot prove speedup |
| HBM nominal bandwidth high, but bottlenecked by 8 B ingress | Identify ingress via bandwidth ledger, implement wide block transfers and outstanding depth; nominal bandwidth cannot enter performance tables |
| FPGA adaptation requires fixed shapes / contiguous / lowered precision | Push restrictions back into board adapter or implement general retiling; preserve original compatibility gates |
| Multicore nondeterminism forced to match on-chip RAM simulation cycle-exact | Preserve legacy B4, protocol trace replay, and HBM statistical calibration separately; do not conflate evidence tiers |
| All switches on is effective, but individual mechanisms yield no gain or short-K regresses | LOO and interaction ablations; select policy or fall back; publish resource overheads alongside negative results |
| Running large after attaching HBM requires shrinking original shapes to pass | Retain actual failures on original large, create independently named development points; do not contaminate frozen suites |
| Increasing cores, cache, or frequency increases E while wall-clock slows | Judge by actual seconds and iso-resource baselines; updating bounds cannot create "free speedup" |

This document defines the R40 contract and implementation trajectory. All commercial mechanism citations draw from public literature; local status is evidenced by code and R39 records; new work packages, tools, thresholds, and on-board results remain to be implemented and verified.
