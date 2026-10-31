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

The GB10 job runs the arms sequentially with at most four CPU cores and a
16 GiB virtual-address limit. Keep both original outputs and report failures.
No result is claimed while the job runs.
