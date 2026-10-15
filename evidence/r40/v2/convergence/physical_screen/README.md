# Physical Implementation Diagnosis

This bundle analyzes existing placement and routing-log evidence for hardware
source `09ebe06`. It does not certify a routed image or change an R40 gate.

`parser_regression.json` reproduces a resource-parser defect in `b81f04d`:
a per-SLR-only table was interpreted as a global utilization table. The
reported 369,666 LUT total was actually SLR0's count. The corrected parser
requires an explicit global `Used` column; per-SLR-only input leaves global
totals unavailable. The independent full utilization report gives 1,090,775
LUTs, matching the sum of the three reported SLR counts.

`diagnosis.json` binds the existing build manifest, full placed utilization
report, implementation log, and analysis source by SHA256. It records:

- CLB occupancy of 97.59%, 96.54%, and 86.54% across SLR0/1/2, with separate
  LUT utilization and unique control-set counts. Unique control sets may
  overlap between regions; their regional counts are not summed globally.
- Estimated SLR1/2 column demand reaching 2.0417 times capacity even though
  aggregate estimated demand is only 71.33% of capacity. Aggregate capacity
  alone cannot exclude a local crossing bottleneck.
- The observed congestion warning, with final routed timing, measured clock,
  candidate area, graph speedup, and acceptance status still unmeasured.

CLB density uses a configurable diagnostic threshold, not an acceptance
criterion. SLL demand is an implementation estimate, separate from placed
connectivity counts. Historical warnings do not establish a final failure.
Observed baseline occupancy is not a proven floor for modified RTL.

Reproduce from the repository root with a fresh output path:

```bash
python3 tools/r40/resources.py --diagnose-placement \
  --util evidence/r40/v2/convergence/placement_followup/full_util_placed_20261015.rpt.gz \
  --implementation-log evidence/r40/v2/convergence/placement_followup/route_initial_20261015.log.gz \
  --build-manifest evidence/r40/v2/convergence/candidate_manifest.json \
  --out /tmp/r40-physical-diagnosis-new.json
python3 tools/r40/resources_tests.py
```

The tests replay actual reports and reject conflicting global counts,
corrupted or missing SLL capacity, invalid build identities, synthesized-only
placement claims, and diagnostic image/timing qualification arguments. They
also check column reordering, unavailable global totals, non-additive control
sets, incomplete later SLL snapshots, and preservation of existing outputs.
The unmodified reference parser is retained in `reference_resources.py.gz`.
