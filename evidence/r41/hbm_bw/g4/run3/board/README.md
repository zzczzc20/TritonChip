# Corrected candidate board results (in progress)

Image SHA-256 `791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09`,
UUID `ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3`; board counter measured 100 MHz.
The author's current [load decision](author_approval.json) and
[433 MHz exception](clock_waiver.json), [load log](program.log), and
[clock capture](board_clock_after_load.json) are retained here. The clock
capture is scoped to the measured 100 MHz DATA clock; the 433 MHz HBM value
is xclbin metadata.
At 2026-10-30 17:32 UTC, identity passed 3/3: add, FP16 16×32×16 matmul,
and tiny Llama decode. Their individual raw records are in
[`identity_results.tgz`](identity_results.tgz); the original 113-point suite
and other delivery stages are still running or pending.

The first identity attempt failed before device execution because the queue
resolved the virtualenv symlink to system Python, which lacked PyTorch:
[`identity_first_failure.log`](identity_first_failure.log). Commit `498b760a`
preserves the virtualenv path; the passing identity above is the subsequent
attempt on the already-loaded image.

The first HBM attempt passed six cases but failed the 16 GiB request clamp:
the striped driver rejected the oversized request before applying the group's
8 GiB limit: [`hbm8_clamp_first_failure.json`](hbm8_clamp_first_failure.json).
Software commit `134b78c2` fixes the clamp. Identity was run before this host-driver change;
its 8 GiB request is unaffected by the changed oversized-request branch.
On the corrected driver, both HBM experiments passed by 17:40 UTC:
[`hbm8_summary.json`](hbm8_summary.json) and
[`hbm8_big_summary.json`](hbm8_big_summary.json), including
[`hbm8_clamp.json`](hbm8_clamp.json),
[`hbm8_bigoffset.json`](hbm8_bigoffset.json), and
[`hbm8_stream.json`](hbm8_stream.json). Passing these functional checks does
not claim that the remaining delivery stages or performance gates have passed.
