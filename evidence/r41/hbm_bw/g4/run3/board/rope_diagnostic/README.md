# Tiny RoPE cycle regression: preliminary diagnosis

The matched point `op.rope/tiny/fp16-b1-h2-s16-d16` passed on both 100 MHz
images. Its three timed serial-grid samples were 3,659/3,554/3,554 cycles on
run 1 and 4,389/4,276/4,262 on the corrected candidate. First-sample kernel
cycles were 801/757/2,101 and 1,009/1,015/2,365 respectively. The measured
point, hashes and samples are in `../original113_comparison.json`.

The generated assemblies copied from the original board bundles are in
`old/` and `new/`. For each of the three kernels, the two `opt.tnir` files are
byte-identical; the `sched.tnir` files differ. In the first two kernels, the old schedules place gather
loads before independent address work; the new schedules place them closer
to their consumers. This is a candidate explanation for reduced overlap, not
a proven cause: both the image and compiler software tree changed.

The old source's `schedule.py` SHA-256 was
`c0ad44d9ea89e019675e461eed57443ca825121dc3b58b2be52bf80b5e29cae2`;
the candidate's is
`33706bae1d170cd44414ad1a235988967e55449552b2b95b1ad2e0ae022d1af8`.
The old exact target lookup did not list `tnca-sim-vrf-v5-x8` and fell back
to `flipflop`; the candidate's longest-prefix lookup selects `banked2` for
`tnca-sim-vrf-v5-x8-g2c4-dp`. These are different latency models. The latter
is the documented banked-register-file model, so reverting it merely to
recover the old measurement is not justified. Validate any scheduling fix
against the actual candidate target and rerun the registered board comparison;
do not change the live software tree during the active batch.
