# XL41 matrix pilot preparation

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../AUTHOR_DIRECTIVE_20261031.md).

The frozen BF16 transpose, GEMM and output-projection points use the registered
shapes, deterministic inputs and tolerances. Standard-Torch pilots check every
output element against an independent CPU reference; the transpose must be
materialized with distinct storage. Host input construction is chunked to
bound memory, without splitting the device operation.

On GB10 CPU all three passed, checking 268,435,456 transpose elements,
3,145,728 GEMM outputs and 248,320 projection outputs. Maximum relative errors
were 0, 0.00294 and 0.00183 respectively. The records and execution log are in
[`cpu_preflight.tgz`](cpu_preflight.tgz), SHA-256
`81ab3c843cb0c572826b3957adf1c994f7eb52342251a3929d63fd8729a12c2d`.
This is CPU evidence only, not a board result or full XL-core acceptance.

The three one-hour board pilots are preregistered in
[`r3-xl41-matrix-queue.json`](../../../../spec/r41/stage2/board/r3-xl41-matrix-queue.json),
SHA-256 `c284b0f6c3bf3bb62fde7864156d7ce62bcf4b64b6500902c6334b4780abd208`.
Their waiter requires the full Q41 core, tail and second-core processes to
pass with verified candidate-image results. It checks card idleness, image,
clock and the shared queue lock and never loads an image. Five prerequisite
validation tests pass ([`waiter_tests.log`](waiter_tests.log)). Gather,
attention and recurrence remain unmeasured and require further work.
