# XL41 GEMV pilot preparation

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../AUTHOR_DIRECTIVE_20261031.md).

The frozen `XL41-gemv` point is BF16 matmul M=1, K=2048, N=6144 with the
registered deterministic inputs and tolerances. A bounded standard-Torch
pilot compares all 6,144 output elements against independent CPU FP32
arithmetic from the rounded inputs, saving both outputs. On GB10 CPU it passed
with zero mismatches and maximum relative error 0.00279:
[`cpu_pilot.tgz`](cpu_pilot.tgz), SHA-256
`8c7c7c3072022134d70abec06cf755fe1343668a86d5f09ac692ac1134023023`.
This is CPU evidence only, not a board result or a full XL-core pass.

The one-hour board pilot is preregistered in
[`r3-xl41-gemv-pilot.json`](../../../../spec/r41/stage2/board/r3-xl41-gemv-pilot.json),
SHA-256 `2c30fa79ba4cb48defc4d1f67596c88daf6ca3a027adc59ce53627db7c609676`.
It uses the same image, clock, shared-card lock and idle checks as the Q41
pilot and never programs the card. Its waiter requires the preceding Q41
pilot to have passed on the candidate and verifies that result's hash and
identity; five evidence-validation tests pass ([`waiter_tests.log`](waiter_tests.log)).
The other seven XL points and full Q41 workloads remain unmeasured.
