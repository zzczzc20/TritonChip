# R40 native cold-call and steady-call audit

The frozen protocol-2 sample plan remains unchanged. Native FakeTensor
constant propagation in the locked Torch version clones real device literals
while compiling. Those `_copy_kernel` launches execute on TNCA, but are not
calls in the generated model wrapper. A cold call can therefore contain
compiler setup and the graph together.

ArchBench preserves the entire cold call, its device cycles, wall time,
launch map and artifact provenance in `cold_call` when a timed run encounters
this case. Its bundle has a separate `cold_call/` directory. It then audits
every warmup and timed call against the generated wrapper's exact launch
count and checks the final numerical output against the original reference.
The primary provenance and per-kernel table describe a verified timed call;
only steady samples contribute to the primary timed median and p95.

A cold count mismatch alone is still `not-measured` for a deterministic run
without timed samples. CPU fallback, numerical failure, no device time or
an extra kernel during any steady call remains a failure. The change does
not filter kernels by name, discard raw cold cost, alter numerical tolerance
or change the required sample count.

Validation uses real standard Torch TNCA graphs. The first add call records
27 device launches, including compiler literal copies; subsequent calls each
record its one generated graph kernel. A negative case executes an actual
additional device graph: its correct output does not bypass the launch-count
refusal. A native cumsum protocol-2 smoke verifies three separate timed
samples and graph/cold provenance separation. These smokes do not establish
any R40 performance threshold or replace the full frozen suite.
