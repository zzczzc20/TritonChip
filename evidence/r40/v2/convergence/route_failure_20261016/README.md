# Three-core implementation: terminal routing failure

Required-gate status is unchanged. The original implementation used hardware
source `09ebe06` and the build directory
`/home/zzhouch/tnca_build/u280_r40ownership_preserved09`.
The source run was not restarted or modified during this investigation.

## Authoritative evidence

- `link.exit` is the actual terminal link exit code: **1**.
- `runme.log.gz` is the complete implementation log, compressed without
  altering its contents. Vivado exited at **2026-10-15 14:53:10 UTC** (22:53:10 HKT).
- `observation.json` records the observation time, actual process listing,
  and path, size and SHA256 of `level0_wrapper_routed_error.dcp`.
  The implementation PID 594866 was absent at observation.

The terminal cause is `Route 35-3`: global congestion level **7**, design not
routable. `Route 35-4445` terminates initial routing. Earlier warnings identify
localized SLL pressure and at least 8,651 CLBs with high pin utilization.
These observations do not establish which individual source change will
resolve the congestion. The retained failure DCP remains on the FPGA server;
its size is 595,473,030 bytes and SHA256 is
`8d82bd375ce489ca433aba15f83692b53b409bc045381a7a8f44ed44811a50d9`.

No final routed timing, CDC qualification, loadable image, board performance,
or original large Torch checkpoint follows from this failed attempt.
Intermediate timing estimates remain diagnostic only.

## Next experiment and decision

The completed read-only report extraction uses `flow/fpga/diagnose_r40_checkpoint.tcl`
against the retained error DCP. The fresh output root is
`~/tnca_build/r40_failed_route_diagnosis_20261016`. It has a 3,600-second wall
limit, four Vivado threads and no synthesis, placement or routing commands.

All reports completed with exit code zero. The complete bundle is
`diagnostic_reports_complete.tar.gz`; `diagnostic_reports_01.tar.gz` preserves
the earlier partial observation. `critical_paths.json` groups the 30 reported
endpoints, while `next_candidate_hypotheses.md` records the next bounded
architecture investigation. Neither bundle qualifies the failed image.

To inspect the completed diagnostic log on the FPGA server:

```sh
tail -30 ~/tnca_build/r40_failed_route_diagnosis_20261016/diagnosis.log
```

Success produces actual routing status, critical paths, hierarchical/SLR
utilization and congestion reports. Use them to select a specific source or
placement hypothesis, then evaluate the smallest causal change. A diagnostic
command failure or timeout preserves partial evidence and triggers a report
repair or competing diagnostic, without restarting implementation.

The [convergence ledger](../README.md) contains every unresolved required gate,
its current blocker, evidence, next experiment and both outcome branches.
The [implementation budget](../implementation_budget.json) permits at most
two additional complete implementation attempts; none has been launched.

`time_metadata.json` records the server’s UTC+8 offset and filesystem UTC
timestamps. An earlier ledger incorrectly labelled the local exit time as UTC;
the corrected UTC time above agrees with the raw log’s modification timestamp.
