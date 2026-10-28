# Full-link bandwidth for memory-bound operators through the standard path

Status: active, from 2026-10-28. Owner: the lead (Claude), working through subagents. This plan replaces the stopped `R41F_PLAN.md` (stream compute unit). **`doc/engineering_discipline.md` is binding and takes precedence over this plan.**

## 1. Target
Memory-bound standard PyTorch operators must reach **32 GB/s of full-link bandwidth on the board**. They run on the three cores of the next image at 100 MHz, through the standard path with default settings.

- **Operators:** `copy`/`clone`, the element-wise binary operators and the unary activations of the registered suite (§3).
- **Full-link bandwidth:** the required logical bytes of the operator, meaning ArchBench `demand` reads plus writes (every input read once, the output written once), divided by the measured device time of the whole operator. The timing boundary is ArchBench's device time, unchanged.
- **Sizes:** the large sizes of the registered suite (tensors of at least 64 MiB).

**Baseline:** the run-1 100 MHz image measured 0.161 GB/s for large fp16 `add` (`evidence/r41/stage2/roofline/points.json`). 32 GB/s is about 200 times that, so an order-of-magnitude gain is the minimum meaningful step.

**What counts:**
- The gain comes from the general mechanisms (discipline §2.2), so every program that moves data through the same mechanisms benefits, not only the suite.
- A gain that depends on recognising the operators, on a non-default setting, or on a unit that only these operators can reach does not count.
- GEMM and other compute-bound operators are not held to the bandwidth number, because they cannot show it. They remain in the non-regression set and must not get slower.

## 2. Where the bandwidth is lost (to be measured, stage by stage)
At 100 MHz, 32 GB/s is 320 bytes per cycle for the chip, about 107 bytes per cycle per core. Every stage of the path must sustain its share, and each stage's ceiling must be measured, not assumed:

1. the core's issue of memory and arithmetic instructions;
2. the load/store unit and the tensor transfer engine (`t.ld.async`/`t.st`) into and out of the vector register file and its windows;
3. the register-file ports;
4. the core's network interface and the mesh NoC;
5. the NUCA home slices (lookup, miss handling, fill, write-back);
6. the AXI outlets;
7. the HBM pseudo-channels and the address layout across them.

Known facts:
- The core memory port is 8 bytes per cycle.
- The tensor transfer engine moves 8-byte beats.
- A home slice serves a hot word in about three cycles.
- Three 256-bit outlets give 9.6 GB/s per direction.

The likely direction is to widen the general data path, with the tensor transfer engine as the streaming mechanism into the register-file windows:
- wider transfer-engine beats;
- the NoC and home-slice width and concurrency;
- streaming, non-allocating NUCA policies through the existing locality controls;
- more or wider outlets, with an interleaved layout across pseudo-channels;
- compiler tiling that uses the transfer engine for element-wise kernels.

The cores keep doing the arithmetic. The data hierarchy stays register file, NUCA, DRAM.

## 3. Registered benchmark (fixed before design work)
A new ArchBench suite, `archbench/suites/r41_membound.json`, registered and committed before any design change is measured. It covers:
- `copy`/`clone`;
- `add`, `sub`, `mul`, `div`;
- `relu`, `silu`, `gelu`, `exp`, `sigmoid`;
- casts among fp32, fp16 and bf16;
- `sum`, `mean`, `max`;
- `layernorm`, `rmsnorm`, `softmax`;
- dtypes fp32, fp16 and bf16;
- small, medium and large sizes, with large at least 64 MiB per tensor;
- contiguous, strided and broadcast layouts.

The baseline is measured on the run-1 image with the same harness and boundary. Every case is reported, and GEMM and the 113 original points form the non-regression set.

## 4. Resources: subtraction first
Before any addition, the per-module resource use of the run-1 routed image is analysed: LUT, FF, BRAM, DSP and CLB packing, per SLR and per module. SLR0 is at 96.74 % CLB packing.

Candidates to remove are redundancy, debug and trace structures, over-provisioned queues and windows, and unused parameter combinations, each with a measured saving and a regression. Removing an architectural capability needs the author's approval (discipline §2.5).

## 5. Workstreams
| ID | Scope | Exit evidence |
|---|---|---|
| G1 | Audit the existing `r41-hbm-bw` and recent `r41` work against the discipline; correct fabricated statements; quarantine constructive exceptions | `evidence/r41/hbm_bw/AUDIT.md` with every finding and its action |
| G2 | Per-module resource analysis of the run-1 image and the removal candidates, with free synthesis screens of the savings | `evidence/r41/hbm_bw/RESOURCES.md` |
| G3 | Register the benchmark suite (§3) and measure every stage's throughput ceiling along the full link for the standard-path kernels, with RTL simulation and SystemC counters plus the board's run-1 records | `archbench/suites/r41_membound.json`; `evidence/r41/hbm_bw/LINK_BUDGET.md` |
| G4 | The general-path design and RTL from G2 and G3, regression, screens | After G2 and G3 |
| E | The lead: integration, the full run on the x86 host, board measurement on approval, final reports | |
