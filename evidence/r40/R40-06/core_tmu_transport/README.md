# TMU control through the existing core instruction path

A small TNCA-IR program compiled through AOT and public `Session` can program
TMU policy and registration registers on the actual three-core MMIO RTL. The
host MMIO window excludes logical address `0x7fff0000`; the core/NI/TMU path
handles it before HBM translation. This establishes a software transport option
without adding a host register aperture or consuming another implementation.

`functional_evidence.tar.gz` retains source snapshots, compiled artifacts,
commands, raw output and reports. The source19 test checks:

- Policy writes, fences and readback for 0, 1, 2, 4 and 34.
- Concurrent range registrations on all three physical cores, repeated with
  the same bases. Each core issues and completes one program in each round.
- The registration counter advances to three and six. Loads of each range's
  final word produce three and six actual tile notices, returning the expected
  zero data through normal memory and ownership paths.
- `nAcc=0xffffffff` keeps every test tile live. The test supplies no speculative
  last-use fact and does not rely on dropping dirty writebacks.

Per-grid telemetry is captured immediately after the registration grid, before
any subsequent policy/readback helper. Each registration grid reports peak
activity three; the final one-program control grid correctly reports one.
Snapshot18 preserves the initial report that sampled only that final grid.
This distinction matters for production diagnostics: an auxiliary control job
is itself a grid and resets the dispatch counters.

The test model uses four sets/two ways and unregistered range arithmetic. It is
an isolated transport diagnostic, not an exact final-image simulation, standard
Torch DCO integration, frozen workload test, or performance result. No required
gate is promoted. Registered-range equivalence has its separate source-bound
proof package; final image, clock and software identities still require their
own qualification.

Next: integrate this control path with standard launch/storage metadata and
correct ownership lifetimes. A policy readback alone is insufficient: runtime
registration and proven liveness must reach the frozen reuse workloads before
DCO or locality measurements can be judged.
