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

Six board HBM cases (window, negatives, samples, alias, crossing, DMA) passed.
Independent 4x16 and 16x16 tensor load/store copies also matched byte for
byte. A diagnostic assembly loaded all-one fragments via vector loads and
ran the matrix operation separately on each physical core. All 256 values on
each core were 12 for DP and 13 for legacy MMA, instead of 16. The raw board
diagnostics and earlier refused diagnostic attempts are in `isolation.tgz`
(SHA-256 `3b0b691e2867f9f42a4404de23eee8fae5f3b734a636a8e3b9c18c724b3ecf63`).

The run-2 RTL declared `MAC_LAT` as an untyped sum of bit-selects. A Vivado
2023.2 synthesis probe with `MAC_CUTS=29` produced zero for that expression
and four for an explicitly `integer`-typed equivalent. The same expression
in Icarus simulation produces four. Thus simulation and synthesis disagree,
which predicts a result-tag pipeline and drain four cycles too short and is
consistent with the board failures. The corrected image still needs board
verification. Raw
Vivado probe source, netlist and log are `vivado_width_probe.tgz` (SHA-256
`0ba5fd5774ffc1791fa58e08c1f71ac93a6721e234764e42ee8fd8b4059b2766`).

The fix explicitly types `MAC_LAT` as an integer. The next image source is
commit `8d237223`, exactly frozen source `1f0a5aff` plus that change. Its
first launch stopped at system link because `unzip` was absent from `PATH`;
no full synthesis or placement occurred. A retry with `unzip` and Vitis HLS
on `PATH` started as `~/r41run3/build_8d237223_retry1` (x86 PID 566798).
It remains the third full synthesis candidate out of the permitted five;
the original run-2 image remains incorrect and is not accepted.

The corrected source passed the RTL `make core-mmadp` test for cut masks 0,
18, 29 and 31, and the capability-off refusal case. A separate Vivado 2023.2
synthesis regression using the corrected declaration verified the output
latency for all 32 masks against popcount. Its raw probe, synthesized netlist
and log are in `vivado_fixed_32_masks.tgz` (SHA-256
`db26ba9d243f57c0eb99e9d7009cf3e960a580dd92a330a952cabb0668451e0c`). These
are RTL and synthesis results, not corrected-image board results.

As a negative RTL control, the same regression was compiled with `MAC_LAT`
forced to zero while retaining `MAC_CUTS=29`; it failed on the first matrix
assertion (`trap code=0x40 id=1`). Icarus exits zero on `$finish` even for this
failure, so the recorded failure is taken from the log, not its exit code.
The passing regression log and negative-control source and log are in
`rtl_fixed_and_bad_latency.tgz` (SHA-256
`3c39ece81ce6b8a5c146f18b788f6e913f1af2939076bb77e08b9df6742b4af3`).
