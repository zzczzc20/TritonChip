# R41 chat speed work

The active-event runtime change and this candidate were developed outside the
fixed board archive. Results below explicitly identify board measurements and
uncalibrated DP SystemC/TLM comparisons.

## Board queue after Qwen3.5 quality failure

The separate Qwen3.5 multi-turn result failed its registered first-turn
comparison; [its evidence](../run3_priority/q35_multiturn_segment_failed_20261031.json)
remains unchanged. The first queue setup [failed before touching the board](board_queue_setup_failure_20261031.log)
because its helper import path was missing; it was corrected. The gated speed
queue then began on the loaded image at 18:49 UTC, but the first event-index session was [killed by its 8 GiB systemd memory
limit](board_event_oom_20261031.tgz); it produced no accepted session result.
The same queue was restarted in a new output directory at 19:01 UTC with a
24 GiB limit after verifying the image, 100 MHz clock and idle card. This is
an operational retry, not a passed measurement.

The [event-index board session](event_index_board_20261031.tgz) completed at
19:40 UTC on the 100 MHz run-2 image; its [summary](event_index_board_20261031.json)
records a full-checkpoint Qwen3.5 session. Teacher and persistent-session
checks passed. Across three rounds, all nine decode steps had zero host reads,
all 924 parameters and four buffers were on TNCA and captured wrappers had no
CPU numerical fallback. The first answer and four token IDs were identical to
the earlier session. For the first call, recorded baseline → event-index
prefill was 0.0331967168 → 0.0344526656 input tokens/s; decode was
0.0142360176 → 0.0157428920 executed steps/s, with 11365085627 → 11320317067
device cycles over three steps. Each decode step had 725 launches and
561/559/559 evictions on both runs. Physical code reload counts were not
recorded on the board. Decode non-device wall residual was 0.4606881353 →
0.4059515709; this includes waits and transfers, not just host execution.
These are single board sessions, not a repeated controlled benchmark. The
quality failure above remains unchanged.

The [equal-row board session](equalrow_board_20261031.tgz) completed at
20:19 UTC; its [summary](equalrow_board_20261031.json) records a full-checkpoint
Qwen3.5 session. Teacher and persistent-session checks passed. All nine decode
steps had zero host reads, all 924 parameters and four buffers were on TNCA,
and captured wrappers had no CPU numerical fallback. First-call output and
all four token IDs matched the event-index session. Event-index → equal-row
prefill was 0.0344526656 → 0.0350916254 input tokens/s and decode was
0.0157428920 → 0.0179191362 executed steps/s. Three decode steps used
11320317067 → 10052258385 device cycles; each had 725 launches and
561/559/559 evictions. Board physical code reloads were not recorded.
Non-device wall residual was 0.4059515709 → 0.3995740425 and includes waits
and transfers. These are single observations, not repeated controlled runs.
The pipeline comparison completed after this result; the separate Qwen3.5
multi-turn quality failure remains unchanged.

The [pipelined-weight board session](pipeline_board_20261031.tgz) completed at
20:55 UTC; its [summary](pipeline_board_20261031.json) records a full-checkpoint
Qwen3.5 session. Teacher and persistent-session checks passed. All nine decode
steps had zero host reads, all 924 parameters and four buffers were on TNCA,
and captured wrappers had no CPU numerical fallback. First-call output and
all four token IDs matched equal-row. Equal-row → pipeline prefill was
0.0350916254 → 0.0351260051 input tokens/s, decode was 0.0179191362 →
0.0180497837 executed steps/s, and three-step device cycles were
10052258385 → 8997855984. Each decode step used 725 launches and
561/559/559 evictions; physical board code reload counts were not recorded.
Non-device wall residual was 0.3995740425 → 0.4586354852 and includes waits
and transfers. These are single board observations, not repeated runs. The
activation-staging comparison completed next; the separate multi-turn failure
remains unchanged.

The [activation-staging board session](activation_equal_board_20261031.tgz)
completed at 21:29 UTC; its [summary](activation_equal_board_20261031.json)
records a full-checkpoint Qwen3.5 session. Teacher and persistent-session
checks passed, with nine decode steps recording zero host reads, 924 parameters
and four buffers on TNCA and no captured CPU numerical fallback. First-call
output and all four token IDs matched the pipeline session. Pipeline →
activation-staging prefill was 0.0351260051 → 0.0354596612 input tokens/s,
decode was 0.0180497837 → 0.0221340569 executed steps/s, and three-step device
cycles were 8997855984 → 7402895275. Each decode step used 725 launches;
evictions were 561/559/559 → 560/559/559 and physical code reloads were not
recorded. Non-device wall residual was 0.4586354852 → 0.4538129830, including
waits and transfers. These are single board observations, not repeated runs.
The opt-in quantisation-hoist comparison completed next; the separate multi-turn
quality failure remains unchanged.

The [opt-in quantisation-hoist board session](hoist_board_20261031.tgz)
completed at 22:05 UTC; its [summary](hoist_board_20261031.json) records a
full-checkpoint Qwen3.5 session. Teacher, persistent-session and independent
placement checks passed; the hoisted quantiser launched 120 times in first
prefill. All nine decode steps had zero host reads, all 924 parameters and
four buffers were on TNCA and captured wrappers had no CPU numerical fallback.
First-call output and all four token IDs matched activation staging.
Activation staging → hoist first prefill was 0.0354596612 → 0.0336052103
input tokens/s and decode was 0.0221340569 → 0.0220064829 executed steps/s;
three-step device cycles were 7402895275 → 7413054385. Each decode step used
725 launches, with evictions 560/559/559 → 558/559/559. Physical board code
reloads were not recorded. Non-device wall residual was 0.4538129830 →
0.4562158185 and includes waits and transfers. These are single observations;
they do not show a hoist speedup. The opt-in setting and cross-platform NaN
bit limitation below remain explicit. The separate grouped multi-turn failure
is unchanged; a whole-prefill diagnostic started next.

## Equal-row lowering

On a DP target the compiler now proves equal row addresses and masks before
sharing a load or row-wise computation; it keeps the shared row in scratch
to control register lifetime. `TNCA_EQUAL_ROWS=0` is the control. The
[matched official fused FP8 results](equalrow_default_20261031.tgz) used
the default division setting, fresh processes and independent exact DP
references. These steady calls each used one launch and had zero code loads
and evictions:

| M×N×K | Instructions/MAC before → after | Device cycles before → after | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.70867919921875 → 0.63916015625 | 161804 → 118469 | 0/256 each |
| 1×250×384 | 0.7098645833333334 → 0.6386770833333333 | 256585 → 195907 | 0/250 each |
| 1×1024×256 | 0.70867919921875 → 0.63916015625 | 536417 → 408542 | 0/1024 each |
| 16×256×256 | 0.047428131103515625 → 0.047428131103515625 | 186575 → 186575 | 0/4096 each |

The all-zero activation also matched bitwise. The FP8 and compiled-graph
suite passed 16/16 and the isolated candidate's 113-point default audit
compared 113 unchanged ([gate records](equalrow_gate_partial_20261031.tgz)).
Four [mask and wide-reduction cases](equalrow_mask_20261031.tgz) with equal
or differing addresses and masks matched bitwise with the optimisation off
and on. An [actual integration-tree TLM run](equalrow_integrated_20261031.tgz)
reproduced the 118469 cycles and 0/256 mismatches, with assembly SHA-256
`f59e9b8010bbffb3a1768119a318643610532deaa85875b69aec093e26ce0f7c`
identical to the isolated final candidate.

The [first integration attempt](equalrow_integrated_failed_20261031.tgz)
failed because `os` was not imported in that source tree; the successful
retry imports it locally. Two [integration audit attempts](equalrow_integration_audit_failed_20261031.tgz)
had 0/113 passing because the copied tree lacked the ISA file and then the
`tools/r38` files, respectively. These are not passing audits. The corrected
[actual integration-tree audit](equalrow_integration_audit_20261031.tgz)
compared 113/113 unchanged, with zero changed or uncompared. The board comparison is recorded above.

## Pipelined weight transfers

The DP X8 path now saves fallback gather indices in scratch, reducing vector
register pressure so the existing next-weight transfer pipeline can be
allocated. Two fallback scratch tiles are alternated: a tile is reused only
after its prior transfer's barrier has completed. This matters for consecutive
noncontiguous N chunks, since a core fence does not wait for asynchronous DMA.
`TNCA_X8_SPILL_LANES=0` is the control.

The [DP TLM records](pipeline_double_20261031.tgz) compare against the
equal-row version above, with default settings and exact DP reference. Steady
calls all used one launch and zero code loads/evictions:

| M×N×K | Instructions/MAC before → after | Device cycles before → after | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.63916015625 → 0.64068603515625 | 118469 → 113675 | 0/256 each |
| 1×250×384 | 0.6386770833333333 → 0.6394895833333333 | 195907 → 190395 | 0/250 each |
| 16×256×256 | 0.047428131103515625 → 0.04752349853515625 | 186575 → 182082 | 0/4096 each |

The 1×7×256 consecutive-wrap case also matched 0/7 bits. The FP8 and
compiled-graph suite passed 16/16, and the 113-point default-code audit
compared 113 unchanged, zero changed or uncompared. The uncalibrated TLM
result does not establish a board speedup; board A/B is pending.

## Experimental E4M3 K32 retiling (not merged)

The DP TLM profile was used for these **diagnostics**, not the board. The
existing chat FP8 fused path already emits K32. A separate worktree explored
padding K16 with negative-zero times positive-one and splitting packed K64
into two ordered K32 commands. [Raw diagnostic records](k32_pad_packed_20261031.tgz)
and the [research patch](k32_research_20261031.patch) are retained. The patch
is based on the experimental worktree and also includes its earlier X8 lane
spill experiment; it is not an integration patch.

For a 16×16×16 generated FP8 dot, the K16 control → padded K32 diagnostic
had 0.0844726563 → 0.1118164063 instructions/MAC and 2774 → 3052 steady
cycles, with one launch, zero steady loads/evictions, and 0/256 bit mismatches
against the DP reference for each. Separate negative-zero, mixed-finite and
NaN/infinity cases also had zero mismatches. Padding slowed this case and the
all-E4M3-K32 change is not integrated; the existing chat fused kernel remains
K32.

A standard Torch `torch._scaled_mm` 16x16x64 test, using a forced 16x16x64
GEMM tile and non-default `TNCA_WINDOW_REUSE=1`, required experimental handling
of packed FP8 row chunks on both sides: K64 baseline used
`TNCA_FP8_PACKED_ROW_CHUNKS=1`, K32 used `TNCA_FP8_PACKED_K32=1`. These are
not unchanged default-code measurements. Each result was compared bitwise
with the independent DP reference.

| Steady route | Instructions/MAC | Device cycles | Launches | Code loads/evictions | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| K64 | 0.0552978515625 | 8355 | 1 | 0/0 | 0/256 |
| K32 + K32 | 0.05908203125 | 8227 | 1 | 0/0 | 0/256 |

The K32 assembly contains two DP K32 commands and no K64 command; the
baseline has one K64. An additional transposed-weight AOT test with nonzero
accumulator compared signed zeros, mixed finite values and NaN/infinity
against the exact DP reference: each case had 0/256 mismatches. The
[raw Torch and special-value runs](k32_torch_20261031.tgz) include compiled
assembly and results. The small cycle difference does not establish a chat
speedup; larger shapes, default-path behavior and the 113-point audit remain
unverified. Earlier forced-K64 Torch compilation without the experimental
row-chunk handling was rejected, so that failure is not presented as a
successful baseline. No board retest is queued for this ungated experiment.

The same forced-tile Torch diagnostic was extended to three boundary shapes;
[raw records](k32_torch_shapes_20261031.tgz) report the second call:

| M×N×K | Instructions/MAC K64 → K32 | Cycles K64 → K32 | Launches | Loads/evictions | Bit mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×16×64 | 0.802734375 → 0.86328125 | 8218 → 8087 | 1 → 1 | 0/0 → 0/0 | 0/16 each |
| 1×17×64 | 4.5900735294 → 4.7352941176 | 21388 → 21245 | 1 → 1 | 0/0 → 0/0 | 0/17 each |
| 16×17×128 | 0.2447150735 → 0.2537913603 | 43652 → 43491 | 1 → 1 | 0/0 → 0/0 | 0/272 each |

These show slightly fewer cycles but more instructions, and do not change the
non-default configuration or acceptance caveats above. The [initial shape
runner log](k32_torch_shape_initial_20261031.log.gz) also preserves its path
collision mistakes and the explicit rejection of a forced K64 tile without
window reuse.

## Explicit precise-division hoist experiment (not merged)

An isolated [compiler patch](precise_div_experiment_20261031.patch) adds
explicit `tl.div_rn` lowering, used by the [diagnostic hoist](bench_precise_hoist_diagnostic_20261031.py)
without globally changing the division setting. The [records](hoist_precise_experiment_20261031.tgz)
include both successes and failures. On M=16, N=2048, K=2048 the default
fused → experimental hoisted path used 0.0328402519 → 0.0204660892
instructions/MAC, 8595658 → 5120824 cycles, 1 → 2 launches and 0 → 0 steady
code loads/evictions. Each produced 0/32768 bit mismatches against the IEEE
DP reference. This is TLM, not board throughput.

A positive-zero sign fix passed 256/256 carefully selected values, including
zero divisors and infinity, in both vector and 16×16 tile tests. However,
a random full-range float32 test still failed 3/256 bits: two very large
denominators yielded zero and one small quotient was off by one bit. The
candidate does not satisfy precise division and is not integrated or queued.
An earlier audit was invalidated by changing its source during execution;
the fixed-source audit and suite were stopped after the random test failed.
Neither is counted as a pass.

A further [scaled-division patch](precise_div_scaled_experiment_20261031.patch)
normalises operands by an exact power of two and fixes the sign of zero
quotients. In [DP TLM](hoist_precise_scaled_20261031.tgz), 2,686,976 sampled
finite float32 divisions and two 256-value special/boundary tests had zero
bit mismatches versus CPU IEEE float32. An intermediate 65536-value sample
failed 23 zero-sign bits before the final sign correction; its log is kept.
On M=16, N=2048, K=2048 the latest hoisted diagnostic has 0/32768 output-bit
mismatches, 0.0210611820 instructions/MAC, 5130958 cycles, two launches,
and zero steady loads/evictions. Its control has 0/32768 mismatches,
0.0328402519 instructions/MAC, 8595658 cycles, one launch and zero steady
loads/evictions. The extra launch can offset this device-cycle saving on the
board. These are samples, not a proof over all float32 inputs. The scaled
patch subsequently [failed](hoist_precise_nan_20261031.tgz) 16/256 negative
NaN sign checks; an intermediate fix failed 8/256 mixed signalling/quiet NaN
payload checks. A [new experimental patch](precise_div_nanfix_experiment_20261031.patch)
passed those 256 payload combinations, 65536 finite random pairs and two
256-value special/tile tests. Its fresh suite and audit later completed 16/16 and 113/113 unchanged. A further [cross-architecture CPU comparison](nan_reference_cross_arch_20261031.json)
of the same 256 NaN/infinity/zero pairs found 16 bit-pattern differences
between the board host x86 CPU and the GB10 CPU. The registered Qwen3.5 and
VL CPU references were generated on GB10 ARM; the latest experimental TLM
division matched that GB10 sample but not an x86 CPU reference for these
pairs. The [reusable NaN TLM check](../../../../tools/r41/chat_speed/test_precise_div_nanpayload_tlm.py)
uses the locked GB10 ARM bit-pattern fixture, independent of the test host's
CPU; its [rerun](hoist_locked_nan_tlm_20261031.log) matched all 256 values.
This is a cross-platform NaN-bit limitation, not an observed model failure. The guarded route is integrated and its completed board session above retains this limitation.

[Boundary comparisons](hoist_precise_boundary_20261031.tgz), fused → hoisted,
all matched the independent reference bitwise and had zero steady code
loads/evictions. Each changed from one to two launches. At 1×7×256 the
instructions/MAC were 28.245536 → 28.457589 and cycles 189841 → 191978;
at 1×250×384 they were 0.475490 → 0.469250 and 162680 → 150132; at
16×250×384 they were 0.047169 → 0.050641 and 274449 → 183816. The N=7
regression means the route must not be applied unconditionally.

A [shape-gated prototype](fp8_hoist_experiment_20261031.py) selects the
hoist only for contiguous dynamic block-FP8 matrices with M≥8, N≥4096 and
K≥2048; other shapes retain the original route. Its [TLM records](hoist_wrapper_tlm_20261031.tgz)
show M=1 at 1×6144×2048 stays on one launch, 14643490 cycles and 0/6144
mismatches, while M=16 at 16×6144×2048 uses two launches, 15390134 cycles
and 0/98304 mismatches. The latter control used one launch, 25209537 cycles
and 0/98304 mismatches. Both routes passed their compiled-graph checks. The
latest precise-division candidate also passed 1,048,576 additional random
finite comparisons and the recorded NaN-payload cases. The [model-route patch](hoist_route_experiment_20261031.patch)
is experimental. The [final protected wrapper and gate records](hoist_final_gate_20261031.tgz)
show its full FP8 suite passed 16/16 and its 113-point default-code audit
compared 113 unchanged; the corresponding [final wrapper source](fp8_hoist_final_experiment_20261031.py)
adds device and shape guards. Direct and compiled-graph tests matched the
reference at M=1 and M=16. The reduced-depth Qwen3.5 model TLM comparison completed below. The
cross-platform NaN bit mismatch above remains a limitation; the board test
completed with the registered GB10 ARM reference and the non-default route
explicitly identified. The opt-in is `TNCA_FP8_HOIST_V1_PATH`, pointing to
pinned finegrained-FP8 revision `48f42b23bb4e0122d4775c6d16753dd2efc7f580`;
the default official route remains selected when it is unset. The [queue registration](hoist_board_queue_20261031.tgz)
SHA-256 is `57f088dd9251bc5402ae87a4eaa63f4f707076d8b0548bf3ddeee100a9a686b9`
and it ran after the four earlier gated speed sessions.

The [one-layer VL model TLM comparison](hoist_model_tlm_20261031.tgz) used
the initial prototype before its final device/shape guard and the completed-
event profiler. Against the activation-staging control, prefill had 28 → 30
launches, 529853540 → 512925718 cycles, 23 → 26 physical loads, 328000 →
481728 uploaded code bytes and 26 → 31 evictions. Its FP8 matmuls plus
quantisers used 0.0472591285 → 0.0375922810 instructions/MAC and 100938341 →
80957691 cycles. Decode stayed at 25 launches, 19 loads and 0.2593053182 FP8
instructions/MAC; full-step cycles were 489017348 → 494381647 and evictions
15 → 13. Both 151936-element logits tensors matched the control bitwise, with
no captured CPU numerical fallback and zero prefill/decode host reads. This is
reduced-depth TLM evidence, not independent CPU quality or board throughput.

The [four-layer Qwen3.5 model TLM comparison](hoist_q35_model_tlm_20261031.tgz)
used that same initial prototype and the activation-staging control. Prefill
had 239 → 254 launches, 2206641939 → 2010384815 cycles, 141 → 157 physical
loads, 1881792 → 2404032 uploaded code bytes, and 139 → 157 evictions; FP8
matmuls plus quantisers used 0.0342451794 → 0.0245982475 instructions/MAC
and 854682364 → 637574561 cycles. Decode stayed at 95 launches, 77 loads,
716224 bytes and 0.2536279297 FP8 instructions/MAC, with full-step cycles
1344816549 → 1364022828 and evictions 77 → 75. Both 248320-element logits
tensors matched bitwise; no captured CPU numerical fallback and zero decode
host reads were recorded. Each prefill recorded 33 compilation-time host reads.
This is reduced-depth TLM, not board throughput or independent CPU quality.

## Experimental frequency-aware code eviction (not merged)

A generic [experimental runtime patch](code_cache_lfu_experiment.patch) ranks
resident code by recent artifact use, aging counts every 256 lookups. The
control is the existing least-recently-used eviction. The
[diagnostic runner](../../../../tools/r41/chat_speed/bench_code_cache_dp.py)
replays the recorded legacy VL one-layer decode load-request order, mapping
its kernels to unique existing DP artifacts by name or closest size. It
loads those artifacts but does **not** execute them; after each sequence it
executes one compiled E4M3 DP dot and checks all 256 FP32 bits against the
independent exact DP reference. Thus this is a synthetic code-pressure test,
not a full-model trace or a board result.

In 11 measured steady repetitions (23 mapped load requests and one dot launch
per repetition), the control → candidate totals were 203 → 187 code loads,
202 → 187 evictions and 2,634,816 → 1,229,888 uploaded code bytes. The dot
had 0 mismatches in all 2,816 output elements for each route, 0.1002197265625
instructions/MAC and 7,271 device cycles per launch for both. The candidate
used non-default `TNCA_CODE_CACHE_POLICY=lfu`; it is not integrated. Raw
results, artifact mapping and [earlier failed setup attempts](code_cache_synthetic_20261031.tgz)
are preserved. The [default-code audit](code_cache_audit_20261031.tgz)
compared 113/113 points unchanged. Subsequent reduced-depth model profiling
found the decode regression recorded below; no full-model speedup or board
acceptance is claimed.

## Unmerged activation and accumulator experiments

Moving activation quantisation outside the N-block programs via the official
separate quantiser and prequantised matmul reduced DP TLM cycles but added a
launch and required non-default `TNCA_IEEE_DIV=1`. These are diagnostic
measurements, not a shipped default. A [default-division check](hoist_default_failure_20261031.tgz)
of the standalone route failed against that reference: M=16, N=256, K=256
had 232/4096 bit mismatches on its first call, with two launches and 110626
cycles. It was stopped and is not accepted. A [follow-up diagnosis](hoist_faithful_diagnosis_20261031.tgz)
compared the same inputs against the bit-exact model of the device reciprocal:
the standalone route matched 0/4096 and the fused route differed 232/4096.
The fused wide lowering selects IEEE division while the standalone default
selects faithful division. This explains the observed difference; the routes
remain unequal and no acceptance follows from the diagnosis. The [raw M=16 run](hoist_m16_tlm.tgz)
and [matched follow-ups](matched_tlm_20261031.tgz) below use the non-default
division setting and compare bitwise with the independent exact reference:

| M×N×K | Instructions/MAC fused → hoisted | Cycles fused → hoisted | Launches | Loads/evictions | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16×256×256 | 0.0474281311 → 0.0433139801 | 186575 → 123126 | 1 → 2 | 0/0 → 0/0 | 0/4096 each |
| 1×256×256 | 0.7086791992 → 0.5082702637 | 161804 → 90331 | 1 → 2 | 0/0 → 0/0 | 0/256 each |
| 16×1024×256 | 0.0474281311 → 0.0359439850 | 598636 → 386256 | 1 → 2 | 0/0 → 0/0 | 0/16384 each |

The nominal 64×64 scaled-mm attempt was split into 16-row passes and did not
reuse weights across all 64 rows. At 64×256×256, it had 0/16384 mismatches but
regressed from 345180 to 404950 cycles, 0.0196266174 to 0.0245447159
instructions/MAC, with one launch and zero steady loads/evictions on both.
A later fallback 64×64 route cost 780226 cycles. Actual multirow reuse was
slower or refused for accumulator/register or code-store capacity; none was
accepted. [Later logs](latest_tlm_20261031.tgz) retain these outcomes and the
uniform-address load experiment. An M=1 spill experiment also regressed from
161804 to 216737 cycles ([record](spill_m1_tlm_20261031.tgz)). Attempts to
issue only one descriptor/fallback DMA were refused by window-lifetime and
barrier verification ([first](branchdma_refused_20261031.tgz),
[second](branchdma_b_refused_20261031.tgz)); the [patch](branchdma_experiment.patch)
is preserved for review and was not shipped.

The candidate's [cache lifecycle unit checks](code_cache_unit_20261031.tgz)
passed 3/3, covering eviction choice, duplicate artifact aliases and adaptation
after a workload phase change. These checks do not replace device or board
validation.

The candidate also passed [16/16 FP8 and compiled-graph DP TLM cases](code_cache_fp8_20261031.tgz),
all bitwise equal to the independent references. The first suite attempt
failed because the offline version mapping was unavailable; the retry pins
the already-recorded Hub revision `0e5764aa5c27d0f005117a92645156481540950a`.
Both attempts and the pinning wrapper are retained.

A [real one-layer Qwen3-VL DP TLM comparison](code_cache_model_tlm_20261031.tgz)
used the same prompt and two generated tokens (`--check-tokens 0`); it is a
reduced-depth performance experiment, not a quality or board result. The
prefill and decode logits each had 0/151936 bit mismatches against the LRU
control, and all 130 launched code texts and launch metadata matched. Both
runs reported no CPU numerical fallback and zero host reads in the measured
prefill and decode steps.

| Step | Device cycles LRU → LFU | Launches | Load marks | Evictions |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 530227895 → 512143370 | 28 → 28 | 23 → 22 | 24 → 22 |
| Decode | 487191706 → 498123453 | 25 → 25 | 19 → 19 | 17 → 20 |

Load marks indicate whether an artifact path was resident, not independently
instrumented physical loads. The profiler's per-launch instruction/cycle
counters were null; the report supplies aggregate step cycles and the
whole-process counter recorded 599723788 instructions for each route. MAC
normalisation and a per-step instruction count are therefore not claimed. The
decode regression and increased evictions mean this candidate is **not
accepted or queued for board retest**. The 16 exact-reference FP8 checks and
113-point default audit above remain distinct from this model comparison.

## Experimental size-aware eviction (not merged)

A second [cache patch](code_cache_sqrt_experiment.patch) scores code by recent
use divided by the square root of its resident size; it is a generic runtime
policy and was tested with non-default `TNCA_CODE_CACHE_POLICY=lfu-sqrt`.
A first-fit [offline trace simulation](code_cache_trace_sim.py), using the
previous one-layer DP run's actual artifact identities and sizes, reproduced
its LRU and LFU load/eviction counts. It estimated LRU → size-aware prefill
loads 23 → 14 and evictions 24 → 18; decode loads 19 → 18 and evictions
17 → 13. These are simulation outputs, not actual model TLM measurements.

In the separate [synthetic DP TLM pressure replay](code_cache_sqrt_tlm_20261031.tgz),
11 steady repetitions with one exact DP dot each had LRU → size-aware loads
203 → 132, evictions 202 → 132 and uploaded code bytes 2,634,816 → 2,145,088.
All 2,816 dot output elements per route matched the independent reference;
instructions/MAC stayed 0.1002197265625, cycles stayed 7271, and probe launch
count stayed one per repetition. The candidate passed 4/4 cache lifecycle
checks and 16/16 fixed-revision FP8/compiled-graph checks. Its
[default-code audit](code_cache_sqrt_audit_20261031.tgz) compared 113/113
points unchanged. The reduced-depth VL and Qwen3.5 comparisons below completed; a corrected
Qwen3.5 counter run is in progress. This policy is not accepted, integrated or
queued for board retest.

The [one-layer VL DP TLM comparison](code_cache_sqrt_model_20261031.tgz)
used the earlier recorded LRU control. Both logits tensors had 0/151936 bit
mismatches against that control and all 130 code texts and launch metadata
matched. Each route reported zero CPU numerical fallback and zero host reads
inside measured prefill/decode. It is reduced-depth and `--check-tokens 0`:
it is not an independent CPU quality check or board performance evidence.

| Step | Device cycles LRU → size-aware | Launches | Load marks | Evictions |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 530227895 → 546235470 | 28 → 28 | 23 → 14 | 24 → 18 |
| Decode | 487191706 → 490524791 | 25 → 25 | 19 → 18 | 17 → 13 |

The load marks are path-residency indicators, not independently counted
physical loads. Whole-process instructions were 599723788 for each; the
profiler's per-launch counters were null, so per-step instructions/MAC is
unavailable. Both measured device-cycle totals increased, despite fewer load
marks and evictions. No end-to-end gain is established; the policy remains
unmerged while the Qwen3.5 comparison runs.

## Asynchronous profiling correction

The previous profile hook read per-launch counters before batched events had
finished, so the model comparisons above correctly left those counters null.
The [updated profiler](../../../../tools/models/decode_profile.py) resolves
completed events after the measured scopes and counts successful code loads
and uploaded bytes directly; compile-only calls are labelled separately.
The report reader excludes skipped calls and identifies warmup separately.

A [DP TLM check](profile_async_check_20261031.tgz) compared all 28 recorded
events to the runtime launch log and verified 0/256 bit mismatches against an
exact FP32 matmul reference. Its two 16×16×16 matmuls each had
0.265869140625 instructions/MAC and took 4916 and 4656 cycles; all 28 events
included three code loads and 13632 code bytes. This is a correctness check,
not a performance comparison. The profile report's scope and legacy-format
checks passed 3/3. Fresh [reduced-depth VL A/B runs](model_profile_and_mask_20261031.tgz)
using the corrected profiler counted actual loads as well as completed events.
For prefill, LRU → size-aware was 28 → 28 launches, 23 → 14 loads,
328000 → 321600 uploaded bytes, 24 → 18 evictions, and 543716965 →
545865198 device cycles. For decode, it was 25 → 25 launches, 19 → 18 loads,
186176 → 185920 uploaded bytes, 17 → 13 evictions, and 499900292 →
505083551 cycles. In the seven FP8 matmuls, prefill was 0.0472591285
instructions/MAC on both routes and 100600246 → 100875903 cycles; decode
was 0.4194615682 instructions/MAC on both routes and 69476943 → 69620129
cycles. MACs are computed from logical activation and FP8 weight shapes.
Both output logits were bit-identical to the LRU route (0/151936 each),
which is not an independent exact CPU reference. The policy remains unmerged.

The same archive preserves the earlier four-layer Qwen3.5 A/B reports and
[comparison script](compare_cache_q35_20261031.py). Both logits have 0/248320
bit mismatches per step against the LRU route, both have zero reported CPU
numerical fallback, and steady decode has zero host reads. Prefill reported
239 launches, 2224324414 → 2206818089 cycles, 141 → 146 load marks and
139 → 149 evictions. Decode reported 95 launches, 1473757373 → 1476651962
cycles, 77 → 54 load marks and 78 → 48 evictions. The marks in this older
profile are not independently counted physical loads. Actual-load and
instruction/MAC profiling is now running with the corrected profiler.

## Equal-row activation staging

The compiler retains its existing equal-row proof through a uniform row
reduction, scalar division, expansion and broadcast, then stages one packed
activation register for repeated rows. The generic implementation is active
for DP and can be disabled with `TNCA_X8_EQUAL_A_STAGE=0` for comparison; it
does not assume a model name. The [initial opt-in experiment](activation_equal_stage_experiment.patch)
and [initial TLM records](activation_equal_stage_initial_20261031.tgz) are
preserved. The integrated source SHA-256 is
`a8ff866ea37cd70551d137b59261424a7a4b7df2f0b929cc11c123f79e993bfb`.

The [gates](activation_equal_stage_gates_20261031.tgz) used the official fused
FP8 kernel, DP TLM and default division. Every shape matched an independent DP
reference bit-for-bit. In each pair there was one steady launch and zero
steady code loads/evictions. The 16/16 FP8/compiled graph checks passed and
113/113 points of the default-code audit were unchanged.

| M × N × K | Instructions/MAC control → candidate | Device cycles control → candidate | Bit mismatches |
| --- | ---: | ---: | ---: |
| 1 × 256 × 256 | 0.640686 → 0.480530 | 113675 → 91280 | 0/256 |
| 1 × 250 × 384 | 0.639490 → 0.475490 | 190395 → 162680 | 0/250 |
| 1 × 1024 × 256 | 0.640686 → 0.480530 | 396516 → 330816 | 0/1024 |
| 1 × 7 × 256 | 31.175223 → 28.245536 | 213645 → 189841 | 0/7 |
| 16 × 256 × 256 | 0.0475235 → 0.0475235 | 182082 → 182082 | 0/4096 |

The zero-input M=1 case also matched all 256 values and had the same recorded
instruction/cycle counts. Five additional [boundary controls](model_profile_and_mask_20261031.tgz)
cover equal and different addresses, partial row masks, differing row scales
and zero inputs, each 0/256 against the DP reference with the option off/on.
Those five custom kernels compiled identically off/on, so they are negative
controls, not proof that the packed path fired. A custom K=128 boundary kernel
was rejected for register exhaustion with the option both off and on; prior
setup attempts also failed. The [failure logs](activation_mask_setup_failures_20261031.tgz)
are retained and are not counted as passes. The official fused K=256 and K=384
cases above did pass. Board A/B testing is queued; these TLM results are not
board throughput claims.

The [one-layer VL model DP TLM comparison](activation_stage_model_tlm_20261031.tgz)
used the corrected completed-event profiler. Control → candidate prefill:
28 → 28 launches, 543716965 → 529853540 cycles, 23 → 23 physical loads,
328000 → 328000 code bytes and 24 → 26 evictions. Decode: 25 → 25 launches,
499900292 → 489017348 cycles, 19 → 19 loads, 186176 → 165184 code bytes
and 17 → 15 evictions. The seven FP8 matmuls had decode instructions/MAC
0.4194615682 → 0.2593053182 and cycles 69476943 → 58568312; prefill
instructions/MAC was 0.0472591285 on both routes. Both model logits tensors
had 0/151936 bit mismatches against the TLM control, no captured CPU numerical
fallback, and zero host reads in prefill and decode. This reduced-depth check
is not independent full-model CPU quality or board throughput evidence.


A [four-layer Qwen3.5 DP TLM activation-stage run](activation_stage_q35_tlm_20261031.tgz)
produced two 248320-element logits tensors bitwise equal to the earlier LRU
TLM control, with no captured CPU numerical fallback and zero steady-decode
host reads. Against that earlier control, prefill had 239 → 239 launches
and 2224324414 → 2206641939 cycles; decode had 95 → 95 launches and
1473757373 → 1344816549 cycles. Both recorded 33 prefill host reads during
compilation. The new completed-event profiler measured 141/77 physical
loads in prefill/decode, while the earlier profiler only recorded load marks;
those two measurements must not be equated. The new run's 25 FP8 decode
matmuls had 0.2536279297 instructions/MAC and 494071298 cycles. A control
using the corrected profiler is still running. These are reduced-depth TLM
observations, not board throughput or full-model CPU-quality results.


The subsequent [corrected-profiler four-layer Qwen3.5 comparison](q35_corrected_model_tlm_20261031.tgz)
replaces the preceding old-profiler control for paired interpretation. LRU →
activation staging prefill: 239 → 239 launches, 2227778949 → 2206641939
cycles, 141 → 141 physical loads, 1881792 → 1881792 code bytes, 139 → 139
evictions; FP8 was 0.0342451794 instructions/MAC on both. Decode: 95 → 95
launches, 1436880921 → 1344816549 cycles, 77 → 77 loads, 905152 → 716224
code bytes and 78 → 77 evictions; FP8 was 0.4137841797 → 0.2536279297
instructions/MAC and 602935166 → 494071298 cycles. Both 248320-element
logits tensors matched bitwise; no captured CPU numerical fallback and zero
decode host reads were recorded. Each prefill recorded 33 host reads during
compilation. This is reduced-depth TLM, not board throughput.

In the same corrected comparison, size-aware cache LRU → candidate prefill
loads were 141 → 146, bytes 1881792 → 2050880 and cycles 2227778949 →
2209153992; decode loads were 77 → 54, bytes 905152 → 885568 and cycles
1436880921 → 1463227902. Launches stayed 239 prefill and 95 decode, FP8
instructions/MAC stayed 0.0342451794 prefill and 0.4137841797 decode,
and both output logits matched bitwise. Decode cycles rose despite fewer
loads, so this policy remains unmerged and is not queued for board testing.

## TNCA-IR AOT compile fix

A descriptor-GEMM experiment exposed an existing optimized `tnca-ir` AOT
compile failure: the optimization pass referenced `imp`, which only exists
in the Triton-import path. The fix uses the normal dead-code pass for the
native IR path. The [gate evidence](ir_aot_fix_gate_20261031.tgz) preserves a
matched baseline failure at optimization level 1 and the fixed DP TLM runs.
The [FP16 descriptor-GEMM test](../../../../compiler/test/test_ir_gemm_tlm.py)
passed 256/256 output bits at each of levels 0, 1 and 2, with one launch and
zero code loads during each launch. The 113-point default-code audit compared
113 unchanged, zero changed or uncompared. These are TLM results, not a board
speed or board correctness claim.
The [short board regression queue](ir_aot_board_queue_20261031.tgz) is
registered after the whole-prefill Qwen3.5 diagnostic. It checks the same
three optimization levels and exact FP16 outputs on the already-loaded image;
source and manifest hashes, the queue lock, image identity, clock and card
idle state are checked before launch. It does not program a new image. Its
result was pending at registration.

## Experimental descriptor E4M3 K32 (not merged)

The [aligned descriptor-GEMM experiment](k32_descriptor_pending_20261031.md)
fixes an existing E4M3 transfer-codegen error and tests K32 for K divisible by
32. The [candidate patch](k32_descriptor_research_20261031.patch.gz),
[TLM records](k32_descriptor_tlm_20261031.tgz) and
[113-point audit](k32_descriptor_audit_20261031.tgz) are retained. Exact DP
outputs matched, but the finite-input cold launches became slower at both
measured shapes. The original source failed to compile; the K16 control first
needed the compiler fixes. Generic tails, partitioned variants, model execution
and board tests are unverified, so this does not establish universal K32 or a
board speedup.

The later [tail and two-role experiments](k32_descriptor_tail_roles_20261031.md)
include exact tested outputs and preserve the failed first two-role harness.
All measured cold two-role K32 comparisons increased visible TLM cycles; the
final [113-point audit](k32_descriptor_v6_audit_20261031.tgz) passed with all code unchanged. Broader model coverage remains pending, and it remains unmerged.

The [TNCA-IR board regression results](ir_aot_board_complete_20261031.tgz)
retain the first attempt, which was killed at an 8 GiB cgroup limit. A
separate registered retry with 24 GiB passed on the same loaded 100 MHz image
without programming: optimization levels 0, 1 and 2 each matched 256/256 FP16
outputs and used one launch, with 2322, 2239 and 2235 measured device cycles.
This is a small regression, not a full chat quality result.

## Experimental 64-lookup cache aging (not merged)

The [experiment and complete TLM records](cache_age64_pending_20261031.md)
include 16/16 FP8 exact cases and the 113/113 unchanged default-code audit.
The matched Qwen3.5 four-layer TLM reduced decode code loads 77 to 52, but
device cycles increased 1344816549 to 1378763635, with unchanged launch count
and exact logits. The candidate remains opt-in, unmerged and not queued for
board testing; these are uncalibrated reduced-model observations.

## Experimental two-step X8 weight lookahead (not merged)

An [isolated candidate patch](x8_depth2_probe_20261101.patch.gz) keeps two
future weight transfers pending, with three separate fallback scratch slots;
the control uses the same source with the new opt-in disabled. The
[DP TLM records](x8_depth2_probe_20261101.tgz) use exact independent FP8
references. Each steady call has one launch and zero code loads or evictions.

| M×N×K | Instructions/MAC control → candidate | Cycles control → candidate | Output mismatches each |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.4805297852 → 0.4800415039 | 91280 → 98871 | 0/256 |
| 1×250×384 | 0.4754895833 → 0.4759895833 | 162680 → 170498 | 0/250 |
| 1×7×256 | 28.2455357143 → 28.2008928571 | 189841 → 188119 | 0/7 |
| 16×256×256 | 0.0475234985 → 0.0474929810 | 182082 → 180079 | 0/4096 |

The two representative decode shapes regressed in device cycles, so this
candidate is not accepted, merged, or queued for board testing. These are
small uncalibrated TLM measurements, not end-to-end throughput. A 113-point
default-code audit was started separately and was pending at this record.

## Experimental equal-row X8 accumulator output (pending gates)

An [isolated opt-in candidate](x8_equal_output_probe_20261101.patch.gz)
uses a single accumulator output row only when all activation rows are proven
equal at every K step; otherwise the original path is retained. The
[paired DP TLM probe](x8_equal_output_probe_20261101.tgz) compares the same
source with the option disabled and enabled against independent exact FP8
references. Each steady case uses one launch and has no code loads/evictions.

| M×N×K | Instructions/MAC control → candidate | Cycles control → candidate | Output mismatches each |
| --- | ---: | ---: | ---: |
| 1×256×256 | 0.4805297852 → 0.41796875 | 91280 → 77837 | 0/256 |
| 1×250×384 | 0.4754895833 → 0.4114270833 | 162680 → 135251 | 0/250 |
| 1×7×256 | 28.2455357143 → 27.1015625 | 189841 → 182169 | 0/7 |
| 16×256×256 | 0.0475234985 → 0.0475234985 | 182082 → 182082 | 0/4096 |

These are small uncalibrated TLM measurements, not board or full-model
throughput. Larger shapes, the 16-case exact suite, 113-point default audit,
and paired four-layer Qwen3.5 TLM profile are running. The candidate is not
yet merged or board-qualified.

The [larger-shape and first proof-attempt records](x8_equal_output_larger_failedproof_20261101.tgz)
show 0/1024 mismatches at 1×1024×256, with steady cycles 330816 → 250578
and instructions/MAC 0.4805297852 → 0.41796875; at 1×2048×2048 they show
0/2048 mismatches, cycles 4735036 → 3359074 and instructions/MAC
0.2644500732 → 0.2018890381. Each uses one launch and no steady code loads.
The first separate masked-row proof harness failed to compile on both paths
because its shape did not enter the X8 lowering; this is recorded as a failed
harness attempt, not a passed correctness case. The gates above remain pending.

The [candidate suite and additional attempts](x8_equal_output_suite_20261101.tgz)
show 16/16 FP8 and compiled-graph tests passing against exact references. A
separate zero-activation 1×256×256 case matched 256/256 on both paths, with
steady cycles 91280 → 77837, instructions/MAC 0.4805297852 → 0.41796875,
one launch and no steady code loads. Two additional proof-harness revisions
failed to compile on both routes: the first had an FP8 constant typing error;
the second exhausted vector registers. Neither is an exactness pass. The
113-point and paired model comparisons are still pending.

The [completed two-step lookahead default audit](x8_depth2_audit_20261101.tgz)
compared 113/113 points unchanged; the candidate remains rejected for the
decode cycle regressions recorded above.

The [smaller proof-harness retry](x8_equal_output_proof_v4_failed_20261101.tgz)
also ran out of vector registers in both control and candidate compilation;
it is not a passing targeted test.

The [smaller exact masked-row test](x8_equal_output_proof_v5_20261101.tgz)
matched 256/256 DP-reference outputs in four input/mask cases for both routes,
but every corresponding assembly hash was identical. It therefore does not
exercise or validate the candidate's changed accumulator-output path.

The [completed default-code audit](x8_equal_output_audit_20261101.tgz)
compared 113/113 points unchanged, with zero changed or uncompared. The paired
four-layer Qwen3.5 TLM comparison is still running; this candidate remains
unmerged and not board-qualified.

A [broadcast-row follow-up](x8_equal_output_proof_v6_20261101.tgz) also
matched the exact reference in four cases, but generated identical control and
candidate assembly throughout; it does not prove the optimized path.

A [one-layer Qwen3.5 profile attempt](x8_equal_output_q35_l1_failed_20261101.tgz)
failed on both routes during cache/mask setup because that reduced model has
no full-attention layer; its comparison also failed after both processes exited.
It provides no model performance result. The four-layer comparison remains
running, and a separate one-layer VL comparison was started.
