# R42: plan, registration and log

> **Binding:** [`engineering_discipline.md`](engineering_discipline.md). Every number below names its domain
> (RTL simulation, SystemC/TLM model, board, analytic). Nothing here is a board result unless it says so.

Branch `R42`, opened from `main` at `4b61abd8` on 2026-11-20. Goal given by the author: implement the reasonable
parts of the scaling blueprint (wider tensor core, line-wide memory path, compiler and runtime work) and obtain a
significant improvement of the ArchBench geometric mean.

## 1. Registration (written before any R42 change was measured)

**Benchmark set.** The frozen ArchBench `tiny` and `small` tiers (`archbench/suites/tiny.json`, 27 points;
`archbench/suites/small.json`, 25 points), every point, through the standard path
(`python -m archbench.run --platform tnca`, default knobs, default optimisation level).

**Measurement domain and boundary.** RTL simulation (`TNCA_BACKEND=sim-file`, Icarus Verilog), device cycles of the
point's whole launch log, as ArchBench records them. No board run is part of this registration: a new image needs
the author's approval and is a later step. Device cycles are compared at equal clock; a frequency claim needs
out-of-context timing evidence of its own and is reported separately.

**Baseline (B42).** `main` at `4b61abd8`, profile `core-vrf-sync-v5-x8` (the per-core profile of the delivered
100 MHz image), default knobs. Measured once on GB10; records under `evidence/r42/baseline/`.

**Candidate.** The `R42` head with the R42 per-core profile and default knobs. A new profile name is used when the
hardware changes; the baseline profile keeps its RTL parameters and its cycles (non-regression gate).

**Metric.** `S_geo = exp(mean_i log(T_B42,i / T_R42,i))` over all registered points of a tier, `T` in device
cycles (PROTOCOL.md §5.1 without the bootstrap, since RTL simulation is deterministic). Reported per tier, with the
per-point table and the per-family geometric means. A point that fails numerically on either side is reported as
failed and makes the tier result "not certified". Every point slower than the baseline by more than 2 % is listed
with its cause.

**Numerics.** Each point's tolerance is the one frozen in its suite definition. No tolerance changes.

**Hardware cost.** Each RTL change records its resource and timing evidence (out-of-context synthesis on the x86
host where a unit changed) before it is counted as deliverable on the U280.

## 2. Triage of the blueprint

(filled in after the inventory; see the log)

## 3. Log
