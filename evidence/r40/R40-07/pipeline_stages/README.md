# Two/three-stage producer/consumer ring checkpoint

Source: `57e977f` plus the ring-stage patch committed with this evidence.
Tests ran in `~/tritonchip-r40-pipeline3` on the ARM jump host, using locked
Torch/Triton and Icarus 12, with `PYTHONPATH=python:compiler/python:tools/asm`
and CUDA hidden for these device diagnostics.

`rctx_stages=2/3` and `TNCA_RCTX_STAGES` now reach the common AOT/JIT lowering,
the JIT cache key and the actual partitioner. The default remains two stages,
absent from default AOT options/provenance. Invalid counts are rejected, and
explicit JIT settings override the environment. Resource checks and fallback
reports remain active; requesting a third stage does not relax ownership.

- `mechanism3.log/json`: option validation 18/18; four real RTL cases pass.
  One-step and carried-pointer kernels remain one-role; eligible odd-K and
  six-program grids execute two-role rings whose artifact reports three stages.
  Odd-K serial/split cycles: 8793/8236; six-program grid: 56776/56168.
  These handwritten Triton cases are mechanism diagnostics, not R40 performance
  measurements. Their gains do not establish the frozen ≥1.2 pipeline gate.
- `contract.log`: default O0/O1/O2 artifacts remain 447/447 byte-identical to
  the independent frozen B39 reconstruction.

The added `native_pipeline_tests.py` audits standard fullgraph Torch mm/addmm,
tail and odd-K, actual ring buffers and role mappings, and zero steady tensor
staging. Complete two/three-stage native results are recorded separately after
the running jobs finish. No passing package JSON is written by this checkpoint.

`make r40-pipeline` runs stage-option, barrier and ring/native diagnostics then
judges the frozen R40-07 evidence. Actual DMA/MMA overlap, full adverse-memory
and fault cases, frozen long-K and matrix performance, automatic policy and LOO
remain mandatory before the package can pass.
