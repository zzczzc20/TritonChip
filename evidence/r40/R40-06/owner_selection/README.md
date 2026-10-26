# DCO Owner Selection: Architecture Pre-screen

This experiment evaluates a combinational owner-selection change before a new
FPGA implementation run. It supports R40-06 and the performance analysis
workflow. It does not amend the frozen contract or promote an acceptance gate.

## Motivation and implementation

The current three-core candidate's synthesized timing report contains a
115-level path from tile 0's `perf_evict1` state to the `ms_wbadr` RAM write
enable. Its data-path delay is 23.110 ns, including 16.120 ns of estimated
routing; OOC WNS is -13.353 ns at a 10 ns target period. The full reference
report is retained as `reference_ooc_timing.rpt.gz`. These are synthesis
estimates, not routed results. They motivate reducing query-dependent serial
selection, but do not prove that owner selection is the linked design's
critical path or the cause of its SLL congestion.

The reference scans tensor entries in ascending index order, repeatedly
selecting the matching entry with the highest base. The candidate computes
membership independently, then rejects each entry if another matching entry
has a higher base or an equal base with a lower index. The surviving payloads
are OR-reduced. Base-to-base comparisons depend only on the shared tensor
configuration, allowing synthesis to share them across independent queries.

The payload, unsigned 64-bit membership arithmetic, zero-sized range rule,
no-match result, and lower-index tie priority remain unchanged. No registers,
protocol changes, or additional execution cycles are introduced. Tables with
fewer than eight entries retain the original scan. Early balanced-tree
prototypes regressed generic depth at small table sizes and were rejected;
their raw results are retained.

## Functional evidence and proof scope

The final compositional proof covers every table size from 1 through 16.
It extracts the actual reference and candidate functions, checks identical
membership expressions and supported uses of address/base/size values, and
replaces membership with an unrestricted eligibility mask. Remaining base
operations are unsigned comparisons and winner-base assignment.

For N bases plus the unsigned zero anchor, at most N+1 distinct ordered values
exist. An order-preserving rank embedding in `ceil(log2(N+1))` bits represents
every possible order and tie while preserving zero. Payloads remain
unrestricted. Equivalence of the extracted selection logic over all such
masks, ranks, and payloads therefore establishes equivalence of the original
selection under these explicitly checked abstraction assumptions.

Yosys lowers processes before constructing the comparison miter and exports
SMT-LIB; Z3 checks whether an output mismatch is satisfiable. A tie-priority
mutation must yield a counterexample. Changes to membership, additional base
arithmetic, and an unrecognized 64-bit declaration must be rejected by the
abstraction guard. All four negative checks pass in the final report.

Unrestricted 64-bit `equiv_simple` attempts at eight and sixteen entries
exceeded the 120-second limit. Splitting output groups and merging shared
logic did not resolve the larger proof. These timeouts remain **unproven** in
the raw records; the compositional SMT proof is reported separately. An early
width-guard mutation exposed an incomplete guard; that failed run is retained
alongside the corrected final proof.

A separate unrestricted 64-bit SMT comparison of the actual eight-entry
functions also proves equivalence, without eligibility or rank abstraction.
Its raw solver input and `unsat` result are retained separately from the
compositional proof and earlier bit-level SAT timeouts. The corresponding
unrestricted sixteen-entry SMT attempt also times out at 120 seconds; its
full-width equivalence remains unproven by that attempt. The sixteen-entry
compositional proof remains the supporting proof for that configuration.

The actual candidate also passes `make nuca-dco`: policies 1, 2, 46, 62, and 74,
with seeds 1 and 7, each issuing 2,000 requests. The reference policy-46/seed-1
run and candidate share trace hash `177f11965447aacd` and identical hit, miss,
merge, writeback, and blocking counts. These are functional RTL checks.

## Mapping evidence and limitations

The retained nine-query generic LUT6 mapping gives:

| Implementation | Longest topological path | Generic LUT6 cells |
|---|---:|---:|
| Reference serial scan | 61 | 24,030 |
| Candidate shared priority comparisons | 30 | 18,675 |

The wrapper uses nine independent addresses sharing one eight-entry tensor
table, corresponding to eight cache-way queries and the current query.
This is a generic logic experiment, not a whole-slice area measurement.
No candidate FPGA frequency, routed utilization, graph speedup, power, or
board result has been measured. These fields remain `null` or `not-measured`.

The live implementation uses the earlier immutable hardware source
`09ebe06`; it is not replaced or restarted by this experiment. Its actual
routed critical paths and congestion locations must determine whether a
subsequent OOC or full implementation run is justified.

## Reproduction

Run from the repository root on the jump host with Yosys and Z3 available:

```bash
git show 09ebe06:rtl/nuca/tnca_nuca_slice.v > /tmp/owner-reference.v
python3 tools/r40/dco_owner_order_proof.py \
  --reference-source /tmp/owner-reference.v \
  --entries 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --out /tmp/owner-proof-new --z3 /path/to/z3
python3 tools/r40/dco_owner_screen.py \
  --reference-source /tmp/owner-reference.v --entries 8 \
  --shared-calls 9 --mapping-only --out /tmp/owner-mapping-new
python3 tools/r40/dco_owner_order_proof.py \
  --reference-source /tmp/owner-reference.v --entries 8 --full-width \
  --out /tmp/owner-full-width-new --z3 /path/to/z3
make nuca-dco
```

Output directories must be fresh. Mapping-only mode explicitly reports that
it has not run equivalence. `--engine sat` retains the original compositional
bit-level checker; a timeout must not be interpreted as equivalence.

`summary.json` binds retained artifacts and source hashes. `raw.tar.gz`
contains the reference/candidate RTL, failed and successful experiments,
solver inputs, process-supervisor records, mutation results, and regression
logs. See the convergence ledger for required-gate blockers and the next
physical experiment.
