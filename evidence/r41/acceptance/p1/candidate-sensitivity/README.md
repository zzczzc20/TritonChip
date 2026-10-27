# Per-candidate SystemC sensitivity coverage and decisions

This is a bounded uncertainty study, not a reconstruction of twelve alternative
bitstreams. `run-v2.sh` registered the probes before execution, and the GB10
systemd unit imposed a one-hour limit and 12 GiB maximum. Each one-factor probe
executes real standard Torch mm/addmm artifacts at aligned and tail shapes,
checking a bitwise independent TNCA-order oracle. Configurations, native model
identities, artifacts, outputs and logs are retained in `runs-v2.tgz` (the earlier M1-domain attempt is preserved as `runs.tgz`). `closeout.py`
requires all registered cases, equal source/artifact/output identities and
successful numeric rows, and derives every reported cycle ratio.

V2 enables the existing core-pipeline model so VRF bank and LSU setup keys
are actually active, and verifies the executed configuration has exactly the
registered one-factor delta. The M1-domain attempt is superseded for this table.

The model cannot represent a shorter combinational path at an unchanged clock
as an execution speedup. A latency perturbation below is an uncertainty probe
for the related execution family, not a claim that the RTL change adds or
removes that latency. The mapping deliberately records these limits. Combined
probes from the earlier archive are superseded for this coverage table.

| Candidate | Probe, relative to the same base | What the study can establish | Decision |
|---|---|---|---|
| F01 stall/enable fanout | issue occupancy +1 | Exposure to a hypothetical issue penalty; no fanout/placement model | Keep physical timing decision; cycle effect inconclusive |
| F02 registered scalar reads/scoreboard | scalar latency +1 | Exposure to scalar dependency latency; not bypass equivalence | Keep RTL miter/physical decision; cycle effect inconclusive |
| F03 FMA balance | vector FP latency +1 | Exposure to FP latency; the actual retiming preserves latency | Do not claim a per-cycle FMA gain |
| F04 writeback/PC enable | vector integer latency +1 | Exposure to ALU/writeback delay; not the actual mux path | Retain physical decision; no isolated speedup |
| F05 VRF scoreboard boundary | VRF bank-conflict latency +1 | Core-pipeline operand-bank sensitivity; does not model physical bank-valid paths | No quantitative physical gain attribution |
| F06 grid admission | dispatch latency +4 | Tests the documented admission-cost scale in this model | Include admission overhead in migration interpretation |
| F07 MSHR service snapshot | lookup latency +1 | Sensitivity to home lookup delay; actual snapshot adds no service state | Retain source/miter/OOC decision; no service-throughput gain claimed |
| F08 XLS FMA regeneration | MMA pipeline latency +1 | Sensitivity to matrix pipeline latency, which actual regeneration preserves | No extra throughput from XLS alone |
| F09 FMA stage balance | vector FP latency +1 | Shared FMA dependency-latency sensitivity; not the internal FMA cut | Inconclusive attribution; use physical and bitwise miter evidence |
| F10 LSU request register | LSU setup latency +1 | Tests request-boundary cost scale; not a full RTL LSU model | Preserve measured RTL latency cost; board residual pending |
| T2 transfer/W-forward | DMA setup latency +1 | Sensitivity to transfer setup cost; not all rank-dependent RTL states | Preserve documented extra rank≥2 cycle; no setup-overlap claim |
| X8 byte transpose | archived gather/X8 capability comparison | Real FP8 consumer model executions with matching point sets | Integrated generic byte-transpose capability; board checks remain identity-bound |

The F03/F08/F09 READMEs explicitly record unchanged model latency keys. The
integration regression records cycle-preserving candidates and the actual
F06/F10 costs; T2 records the rank-dependent extra setup state. Relevant RTL
semantics and physical preflight references remain in each candidate directory,
`integration/README.md`, T2's report and the final run-1 physical archive.
Uncommitted F10 unit archives are not silently treated as raw evidence; the
integrated regression is the retained source of its semantic observations.

The X8 raw archive predates this corrective source. It is reused only as the
candidate's original model sensitivity experiment, with an explicit archive
hash, not as final-source acceptance or as a substitute for full BF16 Qwen.
The unused V3 packed-MAC configurations are excluded from the X8 comparison.

Study outcome: bounded and partly inconclusive. The model probes reveal
sensitivity, zero response or mixed response within their domain; none can
rank these physical retiming candidates on its own. Keep the existing
integration decisions based on semantics and physical evidence. The standard
Torch board comparison remains not measured until the lead executes the
registered matched manifests. No new hardware candidate or full image run is
selected from these proxies.
