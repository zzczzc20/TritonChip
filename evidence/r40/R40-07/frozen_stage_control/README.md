# Frozen pipeline-stage control

`TNCA_PIPELINE=1/2/3` selects the existing producer/consumer ring's buffer count
through both AOT and the standard Triton/Torch compiler. `TNCA_RCTX_STAGES`
remains supported; conflicting environment defaults are diagnosed. An explicit
`rctx_stages` compile option takes precedence. The resolved count participates
in JIT identity and lowering. `TNCA_RCTX_ROLES` remains an independent control:
selecting a buffer count does not silently enable role specialization.
With neither stage environment variable set, the existing two-stage default
remains absent from AOT provenance and preserves the frozen B39 artifacts.

The source-bound archive `functional_evidence.tar.gz` contains two source
snapshots, raw logs, runner commands, exit markers and the frozen B39 reference.
`manifest.json` identifies the current source17 snapshot and its completed
phases. Source13/source15 retains the initial functional run; source17 also
covers explicit JIT options overriding conflicting environment defaults.
All source17 phases completed with exit zero:

- Stage options, cache identity, invalid input and override checks passed.
- Standard native Torch `mm` and `addmm`, at 16×16×48 and 17×31×65, passed for
  each of stages 1, 2 and 3. Generated provenance contains respectively two,
  four and six ring windows, two roles per program, numerical agreement and
  zero tensor staging during measured functional launches.
- The unchanged default path compiled all 447 artifacts byte-identically to
  the independently rebuilt frozen B39 reference.

The earlier direct Triton tests retain ordinary/odd-trip/grid kernels and
unsupported shapes that must stay unspecialized. The archive also retains two
initial environment failures: competing CUDA/TNCA drivers and a simulator
that had not been prepared. Subsequent runs isolate CUDA discovery and prepare
the exact simulator explicitly. No failure log was replaced with a pass.

These are small functional diagnostics. Their simulated cycles and printed
ratios are not R40 performance results. No board-image, 100 MHz, long-K,
64 MiB workload or final-image overlap gate is certified. Required gate status
remains not-measured. Next: qualify attempt01, freeze the resulting software
and image identities, then collect the full frozen long-K and pipeline A/B
with role selection held constant. A failure keeps the original shapes and
records its generated program and first numerical/protocol discrepancy.
