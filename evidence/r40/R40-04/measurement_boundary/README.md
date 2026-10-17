# HBM diagnostic measurement boundary and counter ABI

Required R40 gates remain **not-measured**. This checkpoint fixes existing
diagnostic collection; it does not certify final-image HBM capacity,
bandwidth, model headroom, clock, CDC, or robustness.

## Root cause and correction

The multicore MMIO v2 register mux does not implement legacy C0/C8/D0/D8
AXI counter registers. Its default read value is zero. The inherited
`U50Backend.axi_counters()` consequently reported plausible-looking zeros
instead of unavailable data. The multicore backend now returns `None`;
actual per-outlet counters remain available through opcode 17.

The board diagnostic also downloaded its output inside `_launch_cycles`
before reading traffic counters. That included CPU ownership maintenance
and validation reads in the workload's traffic. Output downloads now happen
after drained graph completion and the post-call counter snapshot.

Timing uses the common device-clock span across all launches. Summed active
kernel cycles are retained separately; missing endpoints do not become a
partial graph measurement. Stream warmups explicitly drain before collection.
Legacy beat counts use their eight-byte beat convention; multicore line
counts use the actual 64-byte line ABI. Missing read latency, write-byte
counters or clock qualification remain unknown. Logical tensor traffic is
separate from physical outlet requests.

The counter interval surrounds the whole call and may include argument or
control traffic outside first/last launch timestamps. Accordingly, the
diagnostic labels its ratios `request_*_bytes_per_graph_cycle`; these are not
certified interface peak bandwidth measurements. No assumed 100 MHz is
converted into a fabricated GB/s result. Formal bandwidth experiments still
require their own matched source/image/clock identity and frozen protocol.

The collector records source/harness/image/contract identity when available,
rejects overwriting an earlier JSON attempt and checks source immutability.
An unknown case cannot silently produce a zero-test PASS. Simulator runs
cannot emit board evidence.

## Retained verification

An unchanged standard Torch `(a+b)*0.5` graph over 4097 elements executed
through three-core, three-outlet MMIO/AXI RTL with stage-2 links, the actual
board wrapper and DMA ownership enabled. Output was numerically correct,
all 257 programs completed, peak physical concurrency was three, and every
kernel reported zero tensor staging.

The old inherited reader was deliberately invoked outside the measured
interval to reproduce its failure against the actual RTL register mux:
all four legacy values were zero while the three outlets recorded
350/284/256 read lines and 83/82/81 write lines. This directly distinguishes
unimplemented registers from a workload with no memory traffic.

Collector boundary checks cover line/beat units, counter reset/source
changes, missing data, differing summed cycles versus graph span, missing
endpoints, and exclusion of CPU reads from collection. Backend option and
physical topology checks also passed. The first collector-unit attempt
failed because plugin discovery was incomplete; importing pinned Triton
before its TNCA driver corrected the test setup. That failed attempt is
retained rather than relabelled.

`raw.tar.gz` contains original logs, exit codes and the execution identity.
`summary.json` and `source_overlay.tar.gz` bind the exact backend, collector,
checkers and counter RTL. The final collector checks and the native RTL
reproduction are separate runs; neither is FPGA execution evidence.
The qualified-image checkpoint and complete native77/HBM14/model runs remain
the next delivery prerequisites. The frozen V archive is unchanged; this
necessary backend correction must be included when freezing the subsequent
final measurement software.
