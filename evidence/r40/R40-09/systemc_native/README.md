# Native SystemC architecture checkpoint

No required R40 gate is promoted. Complete-chip qualification and production
timing remain **not-measured**. This checkpoint establishes a reproducible
native execution foundation; its remaining coverage is explicit in the
[model documentation](../../../../tools/performance_harness/systemc/README.md).

The final validated snapshot is **v7**, executed entirely on the GB10 aarch64
jump host with Accellera SystemC 3.0.2. No model workload or build ran on the
FPGA server. The execution path contains no RTL simulator or Verilator model.

[v7-report.json](v7-report.json) and [v7-raw.tar.gz](v7-raw.tar.gz) retain source,
binary, loaded SystemC-library, configuration, canonical contract-lock and
workload identities; independent RTL captures; response streams; Chrome traces;
native compiler/assembler inputs; negative cases; and experiment logs.
The archive SHA is
`f695b910bfea1ff791a54ad877f3708fb65b7c7491436b6bcb8d8c2fccd24b34`.
The source archive is
`0749e2a237ad211ec3e8fb2559f7c9ab9482d185b0a8a2b8dba179970ff885aa`.

Native execution reads the canonical assembled TNCA machine words. Two
three-dimensional grids exercise all three core workers, shared cache lines,
and a subsequent grid that reads the first grid's results. Product compiler
output exercises masked integer and FP32 vector addition with untouched tail
sentinels. Host checks follow explicit cache writeback/invalidation and memory
write acknowledgement. These are development cases, not full Torch, native77,
HBM14, frozen holdout or physical-image acceptance.

The memory model exercises dirty eviction, miss merging, finite MSHR/target
resources, per-link contention, one/all outlets, multiple PC resources, partial
writes, home atomic ALU operations, visibility and negative mutations.
The two independent randomized RTL request captures agree numerically with a
separate issue-order byte-memory oracle and with TLM replay. Replay retains
fixed RTL arrival times and cannot establish closed-loop chip timing accuracy.

The uncalibrated replay completion errors are material: mean absolute error is
53.10225 and 51.97025 reference cycles for seeds 7 and 29; observed maximum
absolute error is 616 and 373 cycles. These are conditional per-request errors,
not transferable confidence bounds. Source-release slip is zero in these
captures. They do not justify ranking small architectural differences or
authorizing an FPGA implementation by themselves.

Three alternating trace-on/off pairs preserve all simulated counters and
complete response streams. Host overhead ranges from **10.7% to 26.2%**, with
an observed median of **24.9%** on this workload and host. The measurements use
blocking `waitpid` and an independent watchdog, avoiding timeout-wait polling
quantization. This finite sample is an observed range, not a statistical bound,
and it makes no claim about hardware instrumentation overhead.

Earlier [report.json](report.json) and [raw.tar.gz](raw.tar.gz) retain v5.
Its timing harness used a timeout-polling wait and its `contract_sha256` field
contained the contract-document hash instead of the canonical lock hash.
Those records are historical diagnostics and are superseded by v7 for
provenance and host-overhead interpretation; they have not been rewritten.

[failures.tar.gz](failures.tar.gz), bound by [failures.json](failures.json),
retains failed source snapshots, strict C++ build diagnostics and the v6
byte-enable regression. The latter returned selected read bytes where the
architectural NUCA boundary returns a full aligned word; v7 corrects the
adapter while retaining standard TLM byte-enable semantics. The expected
wrong-result mutation remains a failing case in the final validation.

The native executor still lacks multi-role execution, DMA/descriptors/windows,
MMA/accumulators, barriers, complete numerical instruction coverage, RF bank
and port resources, calibrated core timing and the standard Torch adapter.
Router buffering, DCO epochs and detailed HBM timing also remain incomplete.
These omissions prevent complete-chip or industrial qualification. They are
not treated as optional acceptance exclusions.
