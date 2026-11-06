# XL41 >2 GiB stream pilot preparation

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../AUTHOR_DIRECTIVE_20261031.md).

The frozen `XL41-stream` point uses one 2,147,483,776-byte BF16 storage and
a materialized standard-Torch copy. Host input construction is chunked to
bound memory; the device operation remains one full-storage clone. The pilot
checks every element against the registered deterministic input, records full
input/output checksums and the first, last and 2 GiB-boundary values, and
requires distinct device storage and high-address coverage.

On GB10 CPU, the exact-size pilot passed 1,073,741,888 elements with zero
mismatches and identical complete checksums:
[`cpu_pilot.tgz`](cpu_pilot.tgz), SHA-256
`c4d86a1b11cce62dd497331520541606f322510febdf1f0beef2c638ebd85241`.
This is CPU evidence only, not a board result or a full XL-core pass.

The one-hour board pilot is preregistered in
[`r3-xl41-stream-pilot.json`](../../../../spec/r41/stage2/board/r3-xl41-stream-pilot.json),
SHA-256 `b143c2e39a581d8969f8777f9c54a501ea4907c3620ea688fe31da364be0d7c3`.
Its waiter verifies a successful candidate-image XL GEMV pilot before checking
card idleness, image, clock and the shared queue lock. Five prerequisite
validation tests pass ([`waiter_tests.log`](waiter_tests.log)). It never loads
an image. The other XL points and full Q41 workloads remain unmeasured.
