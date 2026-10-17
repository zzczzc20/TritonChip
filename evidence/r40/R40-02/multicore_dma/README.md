# Multicore DMA ownership — complete software/register/AXI functional path

Source: parent `3299288` plus the DMA changes committed with this evidence.
ARM jump host, Icarus 12, locked Torch/Triton. Native run uses the real
three-core U280 AXI-Lite platform RTL stand-in with two-stage NoC/dispatch/home/
code/outlet links, three AXI outlets, and maintenance enabled.

MMIO opcode 18 snapshots a byte range and replies after cache ownership and
physical AXI/link drain. AXI faults refuse admission or fail completion;
range errors, descriptor mutations, stale rings and reset cannot grant a
different range. Completion is latched until the final idle boundary.
R_AXI_STAT bit 4 advertises the implementation. Legacy images lack it and
explicit DMA requests are refused before BO allocation; no coherent-word
fallback is silently substituted.

The common board backend invokes ownership before each BO read/write. The real
board's existing XRT BO methods perform the transfer. The RTL stand-in uses
direct backing-memory byte/word access, selecting the canonical home shard,
with no cache or coherent-MMIO access in that copy. This is a functional BO
model, not an XRT throughput/latency model; its wall time and emulated clocks
are not performance evidence. Aligned word batches avoid artificial per-byte
MMIO delays; unaligned byte access preserves partial ranges.

Retained results:

- `mmio_control.log`: disabled/enabled control cases pass snapshot, delayed
  drain, range refusal, physical faults and reset.
- `guards.log`: 6/6 refused/faulted handoffs perform no BO read/write.
- `options.log`: 13 legacy identity, geometry, capability and allocation-group
  boundaries pass.
- `core1_model.log`, `board3_model.log`: 1-core/1-port/direct and
  3-core/3-port/two-stage real platform paths pass dirty download, clean-line
  invalidation, partial-line dirty-neighbor preservation, zero/range guards.
- `native3_model.log`: unchanged 4097-element standard native fullgraph Torch
  graph, 257 logical programs, three simultaneous active cores, issued/completed
  85/88/84, all three outlets with real read/write traffic, zero steady tensor
  staging, correct CPU reference result.

Runtime DMA is opt-in and requires the capability; currently routing old image
snapshots do not gain it. The final new image must be rebuilt and validated.
New physical placement rules cover the central ownership controller and
per-home command/reply stages, but still require actual new DCP/route checks.
Board XRT measurements, copy/compute overlap on independent regions, timeout/
recovery stress, frozen residency/batch performance and full R40 gates remain
open. No passing package JSON is created by this checkpoint.
