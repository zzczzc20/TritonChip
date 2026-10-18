# Passive TLM socket monitoring checkpoint

Required gate delta: none. This checkpoint supports architectural diagnosis;
it does not qualify a complete-chip model, 100 MHz timing or an R40 workload.
All builds and runs executed on the GB10, outside the FPGA implementation host.

The source archive SHA is
`14a374b2c3781c033e1269b905f018516a5b73c2625a0f2e9c81333b2b728065`.
The compressed raw evidence SHA is
`6326fb2b1ff2ac94daf296dbd1a8f5f2edc3e08764a0eeaa6d9ff4c7d559989a`.
`report.json` retains the campaigns, source identities, recording policy,
independent functional checks and earlier noisy measurements/fixture failures.
`raw.tar.gz` preserves the sealed runs, outputs, RTL capture and mutation.

Standard TLM socket interposers collect bounded aggregate latency histograms,
concurrency, byte counts and optional Chrome transaction events. They insert no
waits or timing annotations. They are not vendor Probe/Monitor Socket classes;
SCV Transaction Recording remains unimplemented. Aggregate mode is the default.

Five alternating off/aggregate/trace samples after warmup observed aggregate
host overhead between -3.01% and +4.93% (median -1.35%). Negative values indicate
host measurement noise, not a speed benefit from instrumentation. Detailed
recording observed +16.65% to +25.59% (median +23.81%). These are workload-specific
observations, not confidence intervals or universal bounds. Simulated cycles,
architectural counters and byte-for-byte responses were identical in all modes.
The earlier, noisier campaign remains in the report.

An independent RTL memory-boundary experiment measured a 36.01x host-time ratio
between the standalone NUCA RTL testbench and fixed-arrival TLM replay. The
frontends perform different work; this is not a native complete-chip speedup.
Conditional completion error averaged 52.57 cycles. No production prediction
error bound or complete-chip speedup has been established. Do not use this
ratio or uncalibrated timing hypotheses to approve the final implementation.

Native execution, descriptor/DMA transfer semantics and role/barrier checks
passed. The retained early-arrive mutation was detected. DMA remains partial:
MMA and complete compiler/runtime workload coverage are outstanding.
The diagnostic harness audits sealed outputs and reports queue/service pressure
and overlapping socket residence without treating them as critical-path shares.
