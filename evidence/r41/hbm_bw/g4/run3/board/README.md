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
[`identity_results.tgz`](identity_results.tgz); later delivery stages are
still running or pending.

The 113-point suite completed with 113/113 passing numeric records:
[`original113_records.tgz`](original113_records.tgz). A preliminary comparison
of matching point hashes and 100 MHz serial-grid cycle medians shows 1.68×
speedup in the sum of point medians; 9 points have larger medians, including
tiny RoPE at 4,276 versus 3,554 cycles (20.3% slower). See all sample counts
and matched values in [`original113_comparison.json`](original113_comparison.json).
This descriptive comparison is not the statistical performance gate; the
slower points need the registered recheck.
The profiles, software trees and compiled artifacts differ between the two
images, so this comparison measures the whole candidate stack and does not
isolate an RTL cause. In tiny RoPE, the three first-sample kernel cycle counts
change from 801/757/2,101 to 1,009/1,015/2,365; the three-sample median
changes from 3,554 to 4,276 cycles. The cause is not yet established.

The suite includes a passing
[`llama_prefill/large`](llama_prefill_large.jsonl) sample with maximum relative
error 0.00237 (tolerance 0.05). Serial-grid device time was 336.820 s,
compared with 582.598 s for the same point on the previous 100 MHz image
([`board_metrics.json`](../../../../acceptance/board_metrics.json)), a 1.73×
device-time speedup. This single comparison does not establish the suite-wide
performance gate.

The default 513-point memory-bound campaign completed 513/513 with no missing
or failed configurations ([stage summary](membound_default_stage.json)). Across all matched 100 MHz points, the sum of
point medians is 1.133× faster and the geometric mean speedup is 1.178×;
81 points are slower, 39 by over 5%, and eight by over 10%.
All point hashes, demand, samples and medians are in
[`membound_default_comparison.json`](membound_default_comparison.json).
This descriptive comparison covers different complete stacks, so it does not
isolate an optimization or replace the registered performance rechecks.
The first tiled batch completed 99/99 with no failed or missing configurations:
[`membound_tiled2_large1.tgz`](membound_tiled2_large1.tgz). Against the same
99 default configurations on the same image and software tree, the sum of
serial-grid cycle medians is 1.742× faster; 15 points are slower, none by more
than 5% ([comparison and samples](membound_tiled2_large1_comparison.json)).
The second large batch passed 72/72
([raw records](membound_tiled2_large2.tgz)). Across both large batches,
171/171 passed; the same-image sum of cycle medians is 1.578× faster, with
31 points slower and none over 5% slower
([all matched samples](membound_tiled2_large_comparison.json)).
This is a descriptive subset, not a full-suite or statistical acceptance.
Both large
batches completed, 171/171 with zero missing or failed configurations; raw
records are in [`membound_default_large1.tgz`](membound_default_large1.tgz)
and [`membound_default_large2.tgz`](membound_default_large2.tgz).
The first medium batch also completed 132/132 with no missing or failed
configurations: [`membound_default_medium1.tgz`](membound_default_medium1.tgz).
The second medium batch completed 39/39 likewise:
[`membound_default_medium2.tgz`](membound_default_medium2.tgz).
The first batch's matching 100 MHz baseline comparison shows 1.134× speedup in
the sum of point medians, with 10 points over 5% slower and two strided BF16/FP16 division
points about 12.6% slower; the point hashes, demands and samples are in
[`membound_default_medium1_comparison.json`](membound_default_medium1_comparison.json).
Across all 171 medium points, the sum of point medians is 1.141× faster,
with 13 points over 5% slower and two over 10% slower:
[`membound_default_medium_comparison.json`](membound_default_medium_comparison.json).
This is a descriptive comparison of different complete stacks, not a
statistical gate or an isolated attribution to an optimization.
For the first large batch only, declared logical read-plus-write bytes divided by
serial-grid device time has a median of 0.134 GB/s and a maximum of 0.544 GB/s:
[`membound_default_large1_rates.json`](membound_default_large1_rates.json).
For all 171 large points, the effective logical-rate median is 0.133 GB/s,
versus 0.138 GB/s on the prior image; the sum of point medians is 1.13× faster
in cycles, with 28 individual points slower. Twelve exceed 5% slower and two
strided BF16/FP16 division points are approximately 12.4% slower. The matched-point data and
samples are in [`membound_default_large_comparison.json`](membound_default_large_comparison.json).
These are effective logical rates, not physical HBM throughput. The tiled
batches remain in progress or pending; this descriptive subset
is not the complete performance gate.

The first small batch completed 142/142 with no missing or failed
configurations: [`membound_default_small1.tgz`](membound_default_small1.tgz).
Its matched 100 MHz comparison has 1.127× speedup in the sum of point medians;
10 points are over 5% slower, including four over 10% slower. The complete
point data and samples are in
[`membound_default_small1_comparison.json`](membound_default_small1_comparison.json).
This is a descriptive comparison of different stacks, not a causal attribution
or a registered recheck. The second small batch completed 29/29:
[`membound_default_small2.tgz`](membound_default_small2.tgz).

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
