# Run-2 board identity failure and diagnostics

The author-approved run-2 image, SHA-256
`1bf769fdbfbeb93586be86561664d43e785af3a01a739d19a3882647132d2d13`,
was loaded after the recorded idle interval. Board UUID was
`01490bbd-d831-23bf-d783-56ec51bd71ce`, and the clock counter read 100 MHz.
The Critical CDC disposition remains open; these are preliminary board
diagnostics, not delivery acceptance.

The default-profile tiny add passed. The tiny FP16 16x32x16 matmul failed
numerically (maximum relative error 1.422, tolerance 0.02), and the cold
call included 26 extra native copy launches. A separate diagnostic verified
both inputs round-trip exactly and reproduced the incorrect result in both a
cold call and a single-launch steady call (maximum relative error 1.475).
The generated matrix instruction was `m.mma.dp.async.k16.f16`.

With all-one matrices the board produced 26 in the first output row rather
than 32; zero input produced zero and an identity-pattern product failed.
For diagnosis only, compiling with the otherwise equivalent profile without
the DP capability also produced 26 rather than 32 with all-one inputs and
failed the identity pattern. This does not establish the root cause. The
run-2 image additionally enables striped HBM, which the earlier passing RTL
board-bench test did not exercise; board HBM diagnostics are in progress.

The raw logs and two original smoke records are in `raw.tgz`, SHA-256
`befbdc1fc7817f8b5493741262df0eb357832a1ea0720cddf33cc6094668e7b4`.
