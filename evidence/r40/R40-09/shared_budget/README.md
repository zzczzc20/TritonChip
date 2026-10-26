# Shared worker admission checkpoint, not accepted

`SharedGraphBudget(path)` is accepted by the existing selector as its budget.
Use one new database path per graph tuning invocation and share it among that
invocation's workers. SQLite IMMEDIATE transactions serialize per-kernel
admission. Reopening preserves the first worker's monotonic start and counts;
a different boot or limit configuration is refused. The limits remain at most
600 seconds per graph and 16 measurements per kernel.

`policy.log` contains 14 passing tests. Four independent processes attempt
64 admissions for the same kernel; exactly 16 succeed. Another kernel has
its own count, while all kernels share the original graph deadline. Reopening
an expired budget refuses a new kernel instead of resetting the deadline.
Existing persistent-cache, device-clock and resource-refusal tests also pass.

This is cooperative admission. Killing an over-budget compile/measurement
worker, automatic compile candidate generation, prediction, frozen holdout
and board performance remain pending. Synthetic test records do not close
R40-09.
