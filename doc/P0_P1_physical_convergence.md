# R40 Physical Convergence Directions (Author Directives, P0/P1 Priorities)

## Executive Summary

The immediate priority is not to "rush into another synthesis run," but to **isolate the nets saturating the SLL hotspot columns and genuinely break up the 90+ stage critical combinational path**. This is the work most likely to reduce delivery turnaround time.

---

## P0: Genuinely Break the tag → owner → victim → writeback Combinational Chain

This is the top-priority candidate to validate in the next RTL iteration, rather than continuing to rely solely on cache parameter reductions.

The repository's existing directions can be advanced directly:

### A. Synchronous Per-Way Tag Banks
- Initiate the tag read with the selected set upon request reception in `E_SEL`.
- Consume the read-back result in the existing `E_LOOK` stage.
- Verify whether the storage actually infers dedicated RAM blocks rather than resolving into flip-flops and large multiplexer trees.
- Specifically validate fills, same-address read/write hazard handling, dirty evictions, maintenance scanning, and upper tag bits.

### B. Victim-Owner Result Registering
- Compute and stage/register the per-way owner in `E_LOOK`.
- Consume the result in `E_MISS`, avoiding packing all classification, dead detection, victim prioritization, and writeback admission into a single cycle.
- Because metadata can change between stages, explicit version checks, re-queries, or equivalent ordering constraints are mandatory; do not simply register stale values and assume they remain valid.

Both items must first undergo **independent functional verification** and **actual TMU + slice screening under identical constraints**. They are currently "well-founded candidate hypotheses," not "proven convergence solutions."

### Mandatory P0 Acceptance Criteria (LUT Reductions Alone Are Insufficient)
1. Exactly at which register boundary was the original long combinational chain severed?
2. What is the new worst path? What are the resulting logic levels and respective logic vs. net delays?
3. What target hardware primitive resources did the tag storage ultimately infer?

---

## P1: Perform Concrete Physical Inspections on Packing and Cross-SLR Pipelining

In the previous failed design, **CLB occupancy reached 97.59% / 96.54% / 86.54%**, which reflects local placement congestion far better than aggregate LUT utilization alone. Do not cite LUT utilization percentages as justification that the design "still fits easily."

Recommended key checks:
- Can critical storage and selection structures **alter their implementation structure** rather than repeatedly slashing capacity?
- Do cross-SLR signals in all three directions (`data`, `valid`, and `ready`) have proper **timing pipeline boundaries**?
- Are physical queue preservation attributes restricted strictly to **essential objects**?
- Are wide-scope preservation attributes preventing the tool from replicating high-fanout registers, re-placing cells, or running physical optimizations?

Note: Do not simply declare "strip all `DONT_TOUCH` attributes"—unintended queue merging in previous runs was a **genuine defect**. Maintain essential boundaries while avoiding over-protection. AMD explicitly notes that preservation attributes restrict subsequent physical optimization; high-fanout registers should be replicated per SLR and per load.

---

## P1: Strictly Separate Correctness Fixes from Current Implementation Identity

The root cause of the latest decoder fix is very specific: when a context exits, younger, killed instructions still retain an illegal opcode classification, contaminating the shared stall / fetch-valid logic and ultimately causing reservation-X. The fix is not to arbitrarily zero out X, but to preserve the illegal flag while preventing illegal instruction classifications from driving incorrect functional unit selection.

However, the **current cf330dc implementation does not contain this fix**. Therefore:
- Maintain a separate parallel track to complete the native four-configuration regression with the fix included.
- **Never cite simulation PASS status on latest software/RTL to endorse an in-flight build running on an older hardware source**.
- Merge all essential correctness fixes before freezing the next implementation candidate.
- If a new RTL baseline is ultimately adopted, corresponding new bitstreams and verification runs are required.

---

## Practices and Anti-Patterns to Avoid

1. **Do not treat connecting all HBM[0:31] pseudo-channels as the primary resource bottleneck**—In failure diagnostics, the kernel hierarchy accounted for 962,157 LUTs, while all four HBM soft interconnect siblings combined totaled only 9,999 LUTs. In terms of LUT footprint, the HBM soft interconnect is negligible. While its placement could still induce routing congestion, this must be demonstrated via cross-SLR net tables rather than preemptively stripping pseudo-channels simply because 32 PCs exist.
2. **Do not equate cache reduction with timing closure**—Reducing 64 to 32 sets represents a legitimate resource trade-off; it does not shrink the benchmark workload and is not an invalid modification. However, it improved local WNS by only ~0.226 ns. Such parameter tweaks must not be presented as evidence that the long combinational path problem has been resolved.
3. **Do not waste the final implementation attempt on directive roulette**—The current candidate preserved `SSI_SpreadSLLs` rather than randomly cycling directives, yet still encountered ~203% local routing demand. Launching the next run with a different directive without modifying wire distribution or critical microarchitectures lacks sufficient rationale.
4. **Do not abort an active run solely due to intermediate WNS, nor wait indefinitely**—The existing detached observer captures diagnostic reports upon completion. Letting the current run finish yields valuable diagnostic data. Concurrently, prepare localized RTL screenings without building redundant launch/observation frameworks or loading large DCPs alongside active placers/routers to avoid memory starvation.

---

## Actionable Execution Checklist (Agent Feedback)

1. **Reprioritize the primary objective to three-SLR physical convergence**; do not treat added features, documentation volume, or test counts as primary indicators of progress.
2. Preserve complete termination logs and DCP checkpoints from the current implementation; **do not automatically retry**, and do not treat intermediate WNS as final results.
3. Add **two diagnostic tables** to the current tooling: (a) cross-SLR bus/module contribution rankings and column-level distributions; (b) deduplicated critical combinational path rankings.
4. Verify key placement details: check whether core-internal logic straddles SLR boundaries, whether global TMU metadata creates high-fanout cross-SLR paths, and where existing pipeline registers actually physically land. Do not rely solely on passing assignment attributes.
5. Prioritize implementing and independently screening the synchronous tag bank and victim-owner stage-register solutions; **prove that the critical combinational chain has been severed, not just that LUT counts dropped**.
6. Maintain exact semantics for metadata commit, dead hints, maintenance operations, and dirty evictions; do not arbitrarily add a pipeline stage to global metadata without resolving visibility.
7. Complete configuration regressions for the decoder fix across all native configurations. Make explicit that the current running image lacks this fix; **strictly prohibit borrowing PASS status across mismatched revisions**.
8. Prior to the next full implementation run, submit **resource usage, worst-case path, logic levels, RAM primitive mapping, and functional validation under identical constraints**, accompanied by concrete rationale targeting SLL hotspots.
9. Do not mask timing violations using false paths, arbitrary multicycle constraints, clock frequency degradation, core reduction, or downscaled frozen workloads. Parametric resource trade-offs must remain consistent across all comparisons.
10. Suspend documentation polishing, non-essential tooling frameworks, and speculative optimizations; maintain essential regressions. Keep physical convergence and correctness on two strictly separate tracks.
11. If localized screening leaves the primary timing bottlenecks unresolved, **report root causes first**; do not initiate a full-chip implementation run simply because one attempt remains in the budget.

**Budget Status**: Exactly one of the two additional full implementation budget attempts has been consumed; one remains. This final run must be reserved for a candidate that has **demonstrated elimination of the primary failure mechanisms**, rather than a candidate submitted merely because "its area is smaller, so let's try again."
