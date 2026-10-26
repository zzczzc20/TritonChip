# Raw device endpoints for candidate and graph timing

Required gate delta: none. Final-image qualification, actual frozen holdout
selection and complete native77/HBM14/model evidence remain open.

ArchBench now retains the existing per-launch device endpoints for every timed
graph call, after all timed calls complete. `audit_graph_samples` recomputes
reported graph cycles and checks the recorded launch count, unsigned counters,
wraps and clock/session/epoch consistency. A missing interior launch is refused
even if the outer interval is unchanged. The ablation reader validates the raw
records whenever their new schema is present; historical records are not given
invented endpoints.

The raw archive contains the unchanged frozen `op.add/tiny/fp16-n1000` point
on three-core/three-outlet board-top RTL with the final candidate cache/TMU
configuration, maintenance and two-stage links. Its reference comparison passes,
with zero steady tensor staging. Two raw timed spans are 2,318 and 2,317 virtual
cycles. These are functional simulation data, not FPGA time or speedup.

The first capture uses the endpoint-retention implementation. A separate,
immutable source snapshot runs the final launch-completeness checker and
regenerates its exact timing entry using the current writer, without another
device run. Both source versions, test logs, raw record, staging identities,
testbench overrides and hashes remain in `raw.tar.gz`. The current pinned ARM
clock tests and the pure protocol checks pass. The default local Python lacks
Torch; that environment failure is not represented as a passing execution.

`noninterference.json` records an AST identity check of the measured `_steady`
loop against source 54249f7. Retention introduces no new device reads or target
instructions. Host postprocessing and serialization cost remain not-measured;
they execute after all timed calls. This is not a board overhead calibration.

The unique downstream blocker is a qualified final image plus the unconnected
standard-path tuning consumer. Next, apply existing bounded workers to the
already compiled fixed 16/32 candidates on the unchanged frozen holdout. Rank
only raw device-event spans; keep compile/tuning costs and formal samples
separate. Success must then produce the complete frozen comparison and regret
records on one immutable identity. Failure retains every refused/expired
candidate and missing point, without changing shapes, thresholds or the contract.
The upstream wall-time benchmark callback is not used by this validation and
must not be enabled for ranking.
