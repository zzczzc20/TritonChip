# Standard-path locality/DCO consumer design

Stage 1 design; runtime integration and board benefit remain Stage 2 work.
No new default policy or environment-only integration is claimed.

## Existing machinery and the missing consumer

`compiler/python/tnca_cc/dco.py` already counts tile reads with bounded
integer interpretation (4,000,000 operations, grid <=4,096 programs), and
emits ordinary scalar stores to the TMU. Its frontend entry takes concrete
argument values and a grid. This is useful analysis, but it does not establish
storage lifetime across a standard Inductor graph. Registering every program
also does not provide a serialized graph-level metadata epoch.

`rtl/nuca/tnca_tmu.v` stages a registration per requester and commits base,
size and descriptor together. `rtl/nuca/tnca_nuca_slice.v` consumes replacement,
dead-block and bypass policy bits. `tools/r40/tmu_control_rtl_tests.py` exercises
that transport through actual common Session kernels. It explicitly does not
certify standard Torch lifetime analysis. Native allocation and counted
leases already exist in `torch_device.DeviceMemory` and `runtime.Session`;
these must remain the ownership authority.

The proposed consumer is a generic Inductor graph preamble, generated from
the scheduler/resource sidecar and resolved against actual native storages
when the graph is bound. It must not call a model-specific numeric kernel.
Data kernels retain their current standard compilation and launch path.

## Information and decisions

| Input or storage class | Consumer action and safety boundary |
|---|---|
| Concrete shape, byte stride, dtype, grid and affine descriptor/pointer reads | Count reads by physical cache line/tile using the existing bounded analysis; preserve masks and padding. Unknown addresses or data-dependent control reject the hint. |
| Views, slices, transposes and tied parameters | Resolve to one storage token, Session and epoch; combine byte intervals and consumers. A different Python tensor is not an independent allocation. |
| Graph input/output or persistent mutable state | Lifetime extends beyond this invocation. Do not declare dead after the graph's last internal read. In-place writes invalidate the previous version's read-count fact. |
| Proven internal temporary | Last use requires all consumers, queued events, asynchronous DMA leases and role handoffs to complete. Python object destruction or source-free alone does not end lifetime. |
| Read-once streaming interval | Tensor bypass is eligible only for proven whole-line accesses, no overlapping reused alias, and no future graph consumer within the policy epoch. Partial-line or unknown masks retain ordinary caching. |
| Reused read-only interval | Keep ordinary caching; bounded nAcc may assist replacement only when all counted readers share the same complete epoch. Counter overflow or TMU-table pressure rejects the hint. |
| Address placement | At initial allocation only, consider legal base-address phases under the actual map. Views keep the storage's address. No relocation of live buffers, hidden repacking or implicit staging. |

The existing `tnca_mem_ports.v` binds each home slice to `home_id % PORTS`.
Changing an outlet count does not change the unique-home mapping or prove
more bandwidth. Placement must account for the actual full address map,
reserved ranges, alignment and fragmentation. A base phase is not a claim
that every line in a contiguous tensor has one home.

## Execution and lifecycle

Before a policy epoch, acquire exclusive graph-level control ownership and
drain earlier work through Session events. Submit a small common TNCA IR
control helper once, then wait for completion before data kernels start:

1. Disable/reset previous TMU metadata at `TMU=0x7fff0000` (reset offset
   `0x18`), using the established transport and an ordinary device fence.
2. Write tile granule (`0x40`) and each range's base/size/descriptor
   (`0x100`, `0x108`, `0x110`); descriptor commits the entry. Encode nAcc,
   bypass, operand and tile granule with the existing emitter's field layout.
3. Write only the selected policy bits at offset zero and fence. Initial
   candidates are replacement and proven read-once tensor bypass; dynamic
   bypass and dead-block prediction need their separate lifetime proof.
4. Execute ordinary compiled kernels. Retain allocation leases and the
   policy epoch until the final dependent event completes. Drain before
   clearing metadata or restoring the prior policy, including error paths.

The helper is control orchestration, not an alternative numerical path.
Its launches/cycles and metadata overhead belong in the A/B graph boundary.
Tuning traffic remains isolated from formal logs; formal control overhead
must not be hidden. Cache identity includes compiler/plan version, graph,
bindings/aliases, strides, placement, epoch, complete image/capability/map
and selected policy. A stale binding or recovery epoch invalidates the plan.

Reject the policy transaction on missing physical TMU capability evidence,
capture without explicit control nodes, overlapping uncontrolled graphs,
unknown alias bounds, atomics/concurrent mutation, unresolved leases,
asynchronous completion uncertainty, allocation reuse or incomplete analysis.
On a device fault propagate Session poison; do not launch a restoration
helper into a lost device. The next explicitly recovered Session starts
with policy disabled and must requalify all metadata.

## Stage 2 board validation

Freeze generic streaming/reuse, alias/mutation, multi-consumer, DMA/role and
long-lived state cases before selection. First prove numeric equivalence,
metadata commit/readback, multi-core exclusion, reset/recovery and the
negative refusals on RTL, then the qualified physical image. Compare the
same resident inputs and code with policy off/on, including preamble and
cleanup overhead. Record actual home/port service, misses, bypass traffic,
cycles, placement and memory use; vary legal base phases separately from
replacement policy. Preserve a no-benefit case. Apply the original default
nonregression and holdout rules before enabling any policy by default.
