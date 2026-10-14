# Loaded common-clock image: native graph correctness

2026-10-14. U280 preflight passed all image/platform/UUID, idle-client, loader
and thermal gates. `program.log.gz` records successful loading of UUID
`50f1225b-ffe2-2271-d855-a77ee4c8f950`. The xclbin SHA256 is the image already
frozen by the parent directory. This is still a one-physical-core image.

`native_clock.log` records actual resident Torch GEMM + reduction through
Inductor/Triton, two dependent kernels per graph, zero measured tensor staging,
and common-clock graph span greater than summed active cycles. Both single and
batched submission pass. Source is software commit `a6dd1e2`; image source and
post-route timing/CDC audits remain those recorded in the parent directory.
This correctness check is not a protocol-2 speedup measurement.

The full existing native inventory has now passed 77/77 (5 warmups, 30 timed
calls, descriptors on, SPMD default enabled). Raw records and the independently
failing device-time performance gate are retained in
`evidence/r40/R40-01/common_clock_board77/`. This smoke does not establish a speedup.
