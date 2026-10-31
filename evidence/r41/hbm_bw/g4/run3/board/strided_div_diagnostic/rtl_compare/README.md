# In-flight LSU sensitivity: registered RTL diagnostic

This isolates `LSU_INFLIGHT=1` versus `2` on the same software/RTL snapshot,
`f7c3a627e3e466205e1717af3cd86283a9391705`, using the registered
`op.div/membound/fp16-small-strided-r32-c1024` point and default compiler
settings. Both images derive from the run-2 image with `HBM_STRIPED=0` because
the RTL testbench uses its small synthetic memory instead of physical striped
HBM; the only difference between these two diagnostic arms is LSU_INFLIGHT.
This is a controlled RTL-model sensitivity, not a board comparison or a claim
that any result explains the measured board regression. The merged RTL and
software snapshot is not identical to the image's frozen source.

The GB10 job ran the arms sequentially with at most four CPU cores and a
16 GiB virtual-address limit. Both passed the numerical check. The one timed
sample was 195,477 cycles with `LSU_INFLIGHT=1` and 123,610 cycles with `2`:
58.14% more cycles for the first arm. The two parameter files differ only in
that setting. The records, parameter files and logs are in
[`results.tgz`](results.tgz), SHA-256
`95ec9ec807069467e976d5c45b136a9ad1afb808dbbedf4617a191d4e7b18de4`.

This is a single-sample RTL-simulation sensitivity with a synthetic AXI
memory and striping disabled. It supports investigating the in-flight limit,
but it neither measures physical HBM nor attributes the observed board
regression to that limit. Do not count its cycle ratio as board throughput.
