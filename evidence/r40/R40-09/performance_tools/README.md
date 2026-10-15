# Performance tooling checkpoint, 2026-10-15

Required R40 gates remain **not-measured** except their previously registered
results. This checkpoint adds diagnostic tools inside existing R40-09. It
does not change the frozen v2 contract or the live FPGA candidate.

Implemented: upstream Proton record effects -> Python TNCA IR -> existing
`s.rdspecial clock` -> deferred epilogue stores -> Chrome trace. Scope names
are unique and nested. Loops inside an outer scope work; branches which skip
or repeat a marker refuse. A dynamic event ring and upstream CUDA collector
are not implemented. The public lowering API is `tnca_cc.proton.lower_records`;
unlowered markers refuse compilation. This is not an automatic Torch profiler.

Each (program, role) owns a fixed buffer slot. Launch grid/context guards
execute before capture writes. Buffers must be freshly zeroed, disjoint and
owned by the caller through completion and CPU handoff. Timestamps remain in
scalar registers until the epilogue: no trace HBM stores inside scopes,
atomics, silent wrapping or hidden register spills. Instrumentation still
changes instructions, live registers, scheduling and shared-unit contention.
Core clocks are independent reset-relative clocks. The viewer's separately
rebased core lanes cannot establish cross-core ordering. An issue timestamp
does not establish DMA/MMA completion without an explicit wait in the scope.

`summary.json` binds raw measurement identities, code, configuration, image,
clock and contract. Its intervals are observed calibration ranges, not a
probabilistic error guarantee. RTL is deterministic here; equal repeats do
not imply zero systematic uncertainty. These are integer-chain diagnostic
microbenchmarks, **not production-workload or board accuracy measurements**.

| Experiment | Baseline cycles | Instrumented cycles | Observed makespan overhead |
|---|---:|---:|---:|
| 128 unrolled additions, fine04 | 147 | 261 | 77.551% |
| 8192 unrolled additions, coarse03 | 8211 | 8325 | 1.388% |
| 128 runtime-loop iterations, loop_short04 | 1935 | 2049 | 5.891% |

Each has six alternating AB/BA pairs, separate warmups, checked output and
device-cycle endpoints. Disabled markers emit exactly the same machine bytes
as the unannotated program. The configurable 1% diagnostic prompt was exceeded;
it is **not a gate, an unusability verdict or a delivery requirement**.
Use the recorded perturbation to choose granularity appropriate to a question.
Whole-program overhead does not bound the error of an individual region.
Region error bounds remain unknown. No constant-overhead subtraction is used.

The initial coarse run's code crossed the default heap boundary; its failure
and correction to the diagnostic harness heap partition are retained. A long
runtime-loop experiment was stopped because simulation calibration time was
disproportionate; its partial warmup and explicit stop reason are retained,
and are excluded from overhead statistics. The bounded loop run then
completed. `r40_proton_dev04_validation.exit` is nonzero for that intentional
stop; independent `r40_proton_dev04_regression.exit` is zero. Do not interpret
the interrupted batch as a passing acceptance run.

Regression: actual pinned Triton 3.8.0 Proton dialect import/nested scopes,
Triton-to-RTL original tests, backend boundary/real reference binding tests,
and compiler/capture-corruption/launch-capacity/calibration refusal checks.
`raw.tar.gz` contains original reports, captures, traces, artifacts and logs.
`source_overlay.tar.gz` reconstructs measured sources over the base commit
in `summary.json`; the final exporter only adds postprocessing uncertainty
metadata. Raw source identities are preserved rather than relabelled HEAD.

Cost model: shared per-PC/direction/NoC/core resource lower bounds can reject
impossible cycle budgets; per-region implementation minima can reject capacity
violations. Neither can approve routability or 100 MHz. Identity-bound
dev/validation calibration refuses frozen holdout, overlapping record IDs,
instrumented training data and extrapolation to a different identity/range.
The old R38 interval's actual-denominator inversion and insufficient-sample
order statistic are corrected. New-model prediction accuracy on real
production dev/validation workloads is **not-measured**. Synthetic calibration
checks verify algebra and refusals only; no R40 holdout was consumed.

Upstream references: [Proton record semantics](https://triton-lang.org/main/dialects/ProtonOps.html),
[official intra-kernel example](https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/example_dsl.py).
The installed 3.8.0 dialect is the actual compatibility test; current main
documentation is not permission to change the project's pinned toolchain.

After the recording/model checkpoint, `tools/performance_harness/` adds the
English agent workflow and a read-only bundle auditor. `harness_audit.json`
replays the preserved fine04 capture and checks its trace, AB/BA raw runs,
code identities and counter partitions. `harness_cost_input.json` asks a
deliberately hypothetical question: can the same binary on the same
single-issue core run twice as fast? Its 136 retired instructions already
exceed that 73.5-cycle budget. This is a resource-bound demonstration on the
recorded microbenchmark, not an R40 speedup target or a production prediction.
The auditor's mutation checks refuse changed trace durations, raw-cycle
disagreements, mismatched code, incomplete buffers and incomplete reports.
