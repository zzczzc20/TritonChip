# Run-3 chat-first campaign

The author-directed v3 VL quality stage started on the corrected run-3 image
at 2026-10-31 03:08:59 UTC. It loaded the checkpoint and opened a board
context, but produced no prompt output and showed no CPU-time progress for
more than nine minutes while kdsstat showed zero outstanding executions.
The process was interrupted at 03:35 UTC so the queued Qwen3.5 session could
start. This is a failed-stalled/interrupted attempt, not a numerical result or
an assertion that the cause is known. The saved process/board snapshot,
experiment report and logs are in [`vl_stalled_attempt.tgz`](vl_stalled_attempt.tgz),
SHA-256 `9b4c070b79e4ac2940a0bfeb51e1190c423c4b1fd0dd5a74fba43b19a96fc2b4`.
The operating system rejected a ptrace attachment; no stack was obtained.

A repeat of the same registered VL quality workload with periodic in-process
Python stack dumps started at 04:22:53 UTC after the priority chats. It does not change the
model or tolerances and is only a diagnostic until actual results are reviewed.
Board-host PID 1151183 required the same image, 100 MHz and an idle board. The diagnostic's registration is
[`r3-vl-stall-trace.json`](../../../../spec/r41/chat/diagnostics/r3-vl-stall-trace.json).

The existing chat implementation explicitly transfers visual outputs to the
host and downloads logits for host-side token selection outside the per-model
transfer meter. Its static `cpu_numerical_fallback=false` flag is therefore
not proof of the author's no-fallback and zero-steady-host-read requirements.
No such requirement is marked passing from that flag.

## Qwen3.5 session, run-2 board

The full-checkpoint session generated `Hi there! How`; its two-token teacher check and persistent-session check passed under the registered v3 rule. The archived raw report, logits, and process logs are in [`q35_session_board.tgz`](q35_session_board.tgz); a machine-readable summary and derived rates are in [`q35_session_summary.json`](q35_session_summary.json). The original quality step was interrupted at 04:22:46 UTC before any prompt result after its host-transfer violation was confirmed; its [preserved attempt](q35_quality_preempted.tgz) is not a quality score. The same registered workload is queued on the device-feedback implementation.

At 100 MHz, the first prefill took 574.5629690899805 s for 15 input tokens, including compilation and output download; the derived rate is 0.026106799092461 tokens/s. The two last decode steps took 131.43974856898421 s in total, 0.015216097274793016 tokens/s, with 723 launches per step and 7,572,841,707 summed device cycles. Subtracting cycles at 100 MHz from wall time gives a 42.385451969839194% residual; it includes host work, transfers, waits and possible counter/overlap effects, not a direct host-only measurement.

This is **not a pass of the author's no-host-fallback rule**: each step downloaded 496,640 bytes of logits outside the model transfer meter and selected its token on the CPU. The old report's `cpu_numerical_fallback: false` was hardcoded. The original result remains preserved; a device-feedback retry is queued.

## VL diagnostic partial output, run-2 board

A live snapshot at 2026-10-31 04:44 UTC records two quality prompts:
`fact` answered `Paris` and `arith` answered `42`, both passing their recorded
reference scores. The full-checkpoint diagnostic remained running at the
snapshot; [`vl_quality_partial_20261031.json`](vl_quality_partial_20261031.json)
is a partial result, not a complete quality run or proof of no host fallback.

## VL diagnostic termination

The traced full-checkpoint VL quality run subsequently exited with SIGSEGV
(return code -11) at 04:49:58 UTC, after the two recorded prompts. The fatal
stack includes the XRT register-write path during device code upload; the
periodic stack dump overlapped the fatal dump, so the root cause is not
established. The [preserved run](vl_traced_sigsegv.tgz) contains the partial
quality result, raw stacks and experiment failure. The board remained bound
to `xocl` with the candidate UUID, and the separately registered device-feedback
VL retry started at 04:50:00 UTC.

## Device-feedback retry and graph reuse

The first device-feedback retry repeatedly recompiled the same model graph:
two captured 7,249-line wrappers differ only in their AOT identifier. The
runtime's `run_and_get_code` resets Dynamo each time, and the original retry
called it on every generation step. The attempt was interrupted at a new
compilation boundary on 2026-10-31 05:26:54 UTC; its logs and both wrappers are
preserved in [`vl_graph_reset_interrupted.tgz`](vl_graph_reset_interrupted.tgz).
This is an interrupted attempt, not a completed chat result.

The replacement captures generated wrappers without resetting Dynamo. A TLM
functional probe returned the expected four tokens, recorded zero reads in the
three decode steps, and recorded new graph counts `[2, 1, 0, 0]`; this is not a
board result. The separately registered replacement VL board stage started at
05:26:56 UTC; Qwen3.5 remains queued behind it.

## Device-feedback VL partial output, run-2 board

The [05:46 UTC snapshot](vl_nohost_partial_20261031.json) records one
full-checkpoint text prompt: `Paris`, passing its reference comparison. The
model lists 822 parameters and four buffers on `tnca:0`; compiled-wrapper
checks for prefill, decode and device selection report no host fallback.
Across 11 executed decode steps, the recorded host-read and staging-download
counts are zero, with 6,215 launches, 25,644,252,382 summed device cycles,
and zero new graphs in each decode step. Output retrieval occurs after the
generation loop and separately records 24 host reads totaling 3,646,560 bytes.

The returned sequence contains two token IDs including EOS, but the device
loop executed 12 steps before EOS was inspected; the cache was then reset.
The recorded 0.023328746064501065 decode steps/s describes those executed steps,
not user-visible tokens/s. Prefill includes compilation. This is one text
prompt only: the image, remaining quality prompts, full session and scripted
chat, and the Qwen3.5 device-feedback run remain outstanding.

The [05:59 UTC snapshot](vl_nohost_partial_two_20261031.json) adds the `arith`
prompt, answered `42` and passed against its reference. For each of the first
two text prompts, every recorded decode step has zero host reads, zero staging
downloads and zero new graphs; compiled-wrapper checks report no host
numerical fallback. Both fixed-length device loops executed 12 steps, so their
step rates must not be presented as user-visible token rates. The complete VL
quality run, image prompt and session checks remain unfinished.

The [06:18 UTC snapshot](vl_nohost_partial_three_20261031.json) records the
third (`reason`) prompt as **failed** under the registered reference rule: the
board output began `We are given the following statements:` and did not contain
`Bob` within 12 tokens; the CPU reference begins `Bob is the shortest.` The
board and reference token sequences differ from the first token. Its recorded
11 decode steps still have zero host reads, zero staging downloads and zero
new graphs, and the wrapper check reports no host numerical fallback. The
quality run continues; the numerical or session cause of the divergence is
not yet established. A separate CPU chunked-prefill diagnostic was started
without changing the registered scoring rule.

A matched full-checkpoint **CPU diagnostic** on GB10 reran only the `reason`
prompt with the same model and 12-token limit. Whole-prompt prefill produced
`Bob is the shortest.`, matching the archived CPU reference; eight-token
chunked prefill produced `We are given the following statements:`, matching
the board response, including all 12 token IDs. The two recorded CPU reports are in
[`vl_reason_cpu_prefill_compare.tgz`](vl_reason_cpu_prefill_compare.tgz).
This identifies a reproducible CPU-side chunking difference; it does not by
itself establish the mechanism or change the failed board quality judgment.

A second full-checkpoint CPU diagnostic compared whole and chunked prefill
using identical 30-token inputs and a 256-token cache. The full and chunked
first-token argmax IDs were 32388 and 1654, respectively; their logits had
relative L2 difference 0.052061811089515686. Cache entries before position 24
matched exactly; differences appeared at or after position 24, where the
chunked path switches from three eight-token prefill calls to one-token decode
calls. The [layer-wise CPU record](vl_prefill_cpu_states.json) supports this
localization, but does not by itself prove why the values differ.

## Grouped short-prefill tail candidate

A generic change keeps the final short prompt block together in a prefill call
instead of treating its tokens as decode steps. The pre-registered prompts,
references, 12-token limit and scoring rule are unchanged. On GB10 CPU, the
VL full-checkpoint diagnostic passed all six quality prompts, including image
and multi-turn; its reason-prompt first-token argmax matched whole prefill,
although logits were not bit-identical (relative L2 0.0329294428229332).
See [`vl_grouped_tail_cpu_diagnostic.tgz`](vl_grouped_tail_cpu_diagnostic.tgz).

For Qwen3.5 on GB10 CPU, the original eight-token chunked path failed the
multi-turn quality check; grouped short tails passed all five quality prompts.
Both full-checkpoint CPU reports are in
[`q35_cpu_tail_compare.tgz`](q35_cpu_tail_compare.tgz). These are CPU
results, not board acceptance. A newly registered run separates quality,
session and VL scripted chat into independently budgeted experiments and
keeps the original scoring rules. The previous board attempt continues until
its current stage ends; the old-code Qwen3.5 follow-up is superseded so the
grouped-tail queue can use the card next.

The [06:56 UTC board snapshot](vl_nohost_partial_four_20261031.json) also
records `instruct` as passed (`OK`), alongside the preserved `reason` failure.
All four completed text prompts report zero host reads and no new graphs in
all decode steps. The next grouped-tail run is already queued; this partial
run is not final acceptance.

The graph-reuse VL quality attempt reached its registered 6,000-second timeout
on 2026-10-31 at about 07:07 UTC. It had recorded four completed prompts: three
passed and `reason` failed. The harness preserved the timeout and SIGKILL as a
failed experiment; its partial `quality.json` still says `running` because the
process was killed. The [raw timeout record](vl_nohost_v3_timeout.tgz) retains
the report, logs, counters and partial output. The grouped-tail VL board quality
stage started at 07:07:01 UTC under the separately registered longer budget;
its results and the queued Qwen3.5 run remain unverified.

## Grouped-tail board attempt, first prompt

The grouped-tail VL quality attempt returned `Paris` on its first prompt but
was marked `failed-no-host-fallback` before quality scoring: its first decode
step recorded 84 host reads (224 bytes). The preserved
[`vl_nohost_v4_failed.tgz`](vl_nohost_v4_failed.tgz) contains the failure and
all per-step counters. The reads were attributed to PyTorch Inductor's
`joint_graph.py:add_node_replacement` constant folding during compilation;
the other ten decode steps recorded zero reads. The model's 822 parameters and
four buffers were on TNCA, and the captured wrappers reported no host
fallback. Compile-time reads and steady execution remain separate observations;
this failed result is not changed. A decode warmup that records compilation
transfers separately and clears its dummy cache is being tested before a new
board attempt. Qwen3.5 started on the board at 07:26 UTC.

The candidate decode warmup passed six local behavior tests on GB10 and a
separate [small TNCA SystemC/TLM model](warmup_tlm_evidence.tgz): generated IDs `[15,15,15,15]`, decode
host-read counts `[0,0,0]`, graph counts `[2,0,0,0]` for the measured steps,
and eight separately recorded post-generation output reads. This simulation
is not a full-checkpoint or board result. The original strict per-step
zero-read gate is unchanged; warmup transfers are recorded separately.

## Qwen3.5 device-feedback board session

The run-2 full-checkpoint session returned `Hi there! How` and passed its
teacher check. It is **failed** under the strict decode-read check: the first
decode step recorded 48 reads (160 bytes) at Inductor constant folding while
compiling; the following two recorded zero. All 924 parameters and four
buffers were on TNCA and captured wrappers reported no host fallback. The
intervening and repeated prompts reported zero decode reads and equal repeat
logits, but the overall persistent-session check was false because the first
generation failed. The quality step did not run. Preserve the complete
[`q35_nohost_v4_failed.tgz`](q35_nohost_v4_failed.tgz).

Board measurement of the first generation, including compilation: prefill
0.025798144433082662 tokens/s, decode 0.006675754800987785 executed steps/s;
4,061 launches and 18,336,258,848 device cycles across its four steps. Its
estimated non-device decode wall residual was 0.7474951188224942; this includes
transfers, waits and possible counter overlap and is not a direct host-only
measurement. The repeated prompt (no new graphs and no decode reads) recorded
prefill 0.10871953129237351 tokens/s and decode 0.014292778234838077 executed
steps/s. These measurements do not pass the full author acceptance.

At 08:04:41 UTC the separately registered warmup VL retry began on the same
run-2 image; Qwen3.5 is queued behind it. No image was reloaded.

## Reason-prompt first-logit CPU diagnostic

With the same full VL checkpoint and 30-token reason prompt, the archived
[CPU logits and diagnostic source](vl_reason_cpu_logits.tgz) show:

| CPU prefill path | Top five token strings (descending) | Top-two gap |
| --- | --- | ---: |
| Whole prompt | Bob, We, To, Let, From | 1.0 |
| Original 8-token chunks and single-token tail | We, Let, To, Bob, From | 1.25 |
| 8-token chunks with grouped short tail | Bob, We, To, Let, From | 0.25 |

The original chunked logits differ from whole-prompt logits by relative L2
0.052061811089515686 (max absolute difference 1.9609375); grouped-tail logits
differ by relative L2 0.0329294428229332 (max 1.125). The grouped top choice
has a narrow 0.25 margin; the original path has a larger change, reproduced
on CPU without TNCA. The previous board output matches the CPU original path's
12 token IDs, but this is not a measurement of board logits and does not rule
out an additional device numerical contribution. A full-prompt DP-profile TLM
comparison is in progress.

A further [full-checkpoint CPU trace](vl_reason_cpu_layer_attention_dp.tgz)
compares the original and grouped-tail paths, using the same first three
8-token chunks. Embeddings and layer 0 outputs are bit-identical. Layer 1 first
differs at position 26; layer 2 first differs at position 24, with the last
position's relative L2 0.011049733497202396, growing to 0.0950397476553917
at layer 27. These are CPU activation comparisons, not TLM or board values.

At layer 2, position 24, CPU Q/K/V projections, the preceding 24-token cache,
rotary inputs, and per-query attention mask match bitwise. The actual SDPA
query and the valid first 25 K/V entries also match, with equal mask, scale and
causal flag; the CPU SDPA output nevertheless differs between query lengths
1 and 6 (max absolute 0.0009765625, relative L2 0.0004917397163808346).
This locates a shape-dependent CPU attention numerical difference; it does
not establish what the board or TLM will produce.

Separately, for the actual position-24 activation of layer 2's Q projection,
CPU mathematical emulation of the ISA DP four-product grouped dot and the CPU
reference path produced identical BF16 outputs for all 2,048 values. Both
differed from BF16-rounded float64 W8A8 exact output at one value (max absolute
0.00390625). This is one projection, not proof for all matrices or a device
result. The full-model DP-profile TLM comparison remains in progress.

A [controlled CPU replay](vl_reason_sdpa_replay.tgz) holds the grouped K/V
cache fixed and compares one versus six query positions with the same mask and
scale. CPU flash SDPA reproduces the position-24 difference (max 0.0009765625,
relative L2 0.0004917397163808346); switching to CPU math SDPA makes the two
outputs identical. The fixed-cache single-query replay also matches the old
cache's result, so these captures isolate query-shape-dependent CPU flash
numerics. No attention backend or acceptance rule was changed. A separate
DP-profile TLM run of the same real position-24 FP8 Q projection is running
alongside the full-model TLM attempt.

The [DP-profile TLM projection run](vl_reason_fp8_dp_tlm.tgz) used the actual
reason-prompt position-24 activation and full checkpoint layer-2 Q-projection
weights. All 2,048 output values matched both CPU W8A8 reference and the CPU
DP-order rederivation bitwise. The TLM identity records `mtx_dp=1`; its three
cores counted 2,560, 3,072 and 2,560 DP commands. This is a single FP8
projection on the SystemC/TLM model, not board evidence or a full-model logits
comparison. The full-model attempt continues.

## Warmup retry: first board prompt

The [partial warmup-retry board record](vl_nohost_v5_first_prompt.json) reports
`fact` passed with `Paris`; the full quality suite is still running. All 822
parameters and four buffers are on TNCA, the captured decode, prefill and
selection wrappers report no host fallback, and all eleven decode steps report
zero host reads and no new graphs. The separately reported decode warmup
recorded 84 compiler constant-folding host reads (224 bytes); prefill recorded
168, and post-generation output retrieval is also reported separately. These
are not hidden by the steady-decode counter.

For this board prompt, prefill was 0.04390292815386181 tokens/s and decode
0.023130249739335265 executed steps/s; the fixed loop executed 12 generation
steps but returned two token IDs including EOS, so the decode-step rate is not
user-visible token throughput. The returned-ID rate was
0.0021479550200169235/s over generation wall time, excluding output retrieval.
The measured non-device decode wall residual was 0.46064795638806766, including
transfers, waits and possible overlap, not a direct host-only measurement.

A [standalone DP-profile TLM attention replay](vl_reason_attention_dp_tlm.tgz)
uses the saved CPU attention inputs from the layer trace above. Compiled wrapper
checks report no host fallback. The one-query TLM result differs from CPU
flash SDPA by relative L2 0.0017929725581780076 (max 0.001953125), but matches
CPU math SDPA bitwise for all 2,048 values. The six-query result differs from
CPU flash by relative L2 0.0015059992438182235; against CPU math, 9 of 12,288
values differ (max 0.00048828125). Comparing the first query across the two
TLM shapes shows one differing value out of 2,048 (max 4.76837158203125e-7).
This is an isolated attention replay, not full-model logits or a board result.

A full-model [CPU math-attention diagnostic](vl_reason_cpu_math_attention.tgz)
used the same prompt and checkpoint with PyTorch's math SDPA explicitly
selected for diagnosis. Its whole-prompt, legacy-chunked and grouped-tail
logits are bitwise identical and each selects `We`, not the archived CPU
flash-reference answer `Bob`. Compared with the original full-prompt CPU flash
logits, the math result has relative L2 0.11928296089172363 and max absolute
6.0: `Bob` changes from 38.25 to 32.25 and `We` from 37.25 to 38.75. This
is a substantial whole-model CPU numerical effect, not merely a tie, and
warns that fixing the prompt tail alone may not fix board quality. The math
backend was used only for diagnosis; the registered CPU reference and scoring
rule remain unchanged. The isolated TLM attention match to CPU math is not a
full-model TLM or board result.

An exploratory [CPU-only attention candidate](vl_reason_cpu_flashapprox_failed.tgz)
rounded the unnormalized softmax weights to BF16 before the value product,
following a visible feature of the pinned upstream
[CPU flash implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp).
This brought grouped-tail logits closer to the flash reference (relative L2
0.05607813224196434 versus 0.12522464990615845 for CPU math), but still chose
`We` rather than `Bob`; it did not fix the registered reason prompt. It was a
CPU diagnostic only, is not a product change, and did not alter the reference,
quality rule, board run, or existing failures.

A second [CPU-only attention candidate](vl_reason_cpu_flashsve_failed.tgz)
re-derived the SVE flash exponential and the BF16 unnormalized weight cast;
it matched the captured six-query layer-2 attention sample bitwise. It still
failed to reproduce the full-model reference: grouped-tail relative L2 was
0.05127841234207153 and its first token was `We`, while the original CPU flash
reference selected `Bob`. This candidate was not integrated or sent to the
board; its failed result and unchanged registered reference are preserved.

The [candidate's CPU full-model trace](vl_reason_cpu_flashsve_trace.tgz)
compared each attention call with the original CPU flash operation using the
same inputs. Its first mismatch appeared in layer 7 of the first 8-token
block: one of 16,384 values differed, with maximum absolute difference
0.0001220703125. The candidate's final first token remained `We`. This
localizes a numerical discrepancy in the candidate; it does not establish that
the single mismatch alone caused the final token change.

An [untimed functional TLM check](vl_reason_fp8_untimed_tlm.tgz) tried the same
DP Q projection with `memory_model=1` and `TNCA_TLM_UNTIMED_COLD=1`; its 2,048
outputs also matched the CPU reference bitwise. Its recorded call wall time was
21.97075816616416 seconds (the separately run timed TLM call recorded
25.601286967284977 seconds). These are individual diagnostic wall times, not
board performance, and the settings differ; no speedup is claimed. The
original full-model timed TLM attempt continues without modification.

A [CPU DP-order rederivation over all seven layer-2 FP8 projections](vl_reason_fp8_all_cpu.tgz)
used the actual position-24 inputs from the legacy reason prompt. With IEEE
activation division in both paths, all 20,480 BF16 output values matched the
CPU W8A8 reference bitwise. Against BF16-rounded float64 W8A8 exact results,
Q projection differed in 1/2,048 values (maximum 0.00390625) and the MLP up
projection in 3/6,144 (maximum 0.0009765625); the other five matched. This is
CPU mathematical emulation on seven projections, not a device or full-model
claim. The Q projection alone was also run on TLM as recorded above.

The [three-prompt warmup-retry board snapshot](vl_nohost_v5_three_prompts.json)
now records `fact`, `arith`, and `reason` as passed. The reason answer was
`Bob is the shortest.  \n\nHere's the reasoning:  \n`; all 12 token IDs
match the registered CPU reference. All three prompts record 822 TNCA
parameters and four TNCA buffers, no captured-wrapper numerical fallback, and
zero host reads and zero new graphs in every decode step. The observed board
reason result supersedes no prior failure; it shows that the CPU math
experiment alone could not predict this grouped-tail board result. Image,
remaining prompts, session and scripted chat are pending.

For the reason prompt on the board, prefill was 0.07936876882250843 tokens/s;
11 decode steps were 0.016516948675659057 executed steps/s, with 6,215 decode
launches and 25,675,912,458 decode device cycles. All 12 generated token IDs
were returned; the generation-wall returned-ID rate was
0.011494638196582864/s, excluding output retrieval. The estimated non-device
decode wall residual was 0.6144657015731809, including transfers, waits and
possible overlap; it is not direct host-only time.

A [two-layer DP TLM replay](vl_reason_layer23_dp_tlm.tgz) used the same full
30-token prompt, real checkpoint, CPU-captured layer-2 input, and one
teacher-forced decode step. The CPU pair reproduced the full-model reference.
On TLM, prefill layer-2 and layer-3 output relative L2 values were
0.00016276411589986764 and 0.00025421574570036655. The layer-2 value cache
relative L2 was 0.010808053152941194 at prefill and 0.010873686087227104 at
decode; both exceeded this diagnostic's available 0.01 criterion. The overall
pair diagnostic was marked failed. Its CPU-Inductor floor could not be
measured because compilation emitted unsupported `vf.cvt.e4m3.f64`; the other
recorded floors are present in the artifact. This is a two-layer TLM result,
not a full-model logit or board failure. A targeted TLM replay of the layer-2
value projection is running to investigate it.

The [standalone V-projection TLM replay](vl_reason_vproj_dp_tlm.tgz) with
CPU-captured full-prompt activations matched the CPU W8A8 and DP-order
references bitwise in both 30-token prefill and one-token modes. This replay
used a no-cache CPU capture; the separate two-layer experiment used a static
cache, so the input identities are not assumed identical.

A [static-cache layer-2 normalization diagnostic](vl_reason_norm_amplification.tgz)
replayed the CPU-captured 30-token input through the standalone norm on DP
TLM: 15,990/61,440 BF16 values differed from eager CPU, relative L2
0.0029610961209982634 (max 0.03125), with no compiled-wrapper fallback.
Feeding the TLM and CPU norm outputs into the same CPU W8A8 V projection gave
relative L2 0.010808045975863934, closely matching the two-layer TLM value
cache's 0.010808053152941194. This supports normalization error amplified by
FP8 quantization as an explanation of that particular TLM cache discrepancy;
it does not prove the board's full-model numerics or change its passing reason
result.

The [four-prompt board snapshot](vl_nohost_v5_four_prompts.json) adds the
`instruct` answer `OK` as passed. All four completed prompts report zero host
reads in their 11 decode steps. The remaining quality prompts, image, session
and script are still pending.

For the same static-cache norm input, [CPU Inductor versus eager CPU](vl_reason_norm_cpu_inductor.tgz)
differed in 15,989/61,440 values (relative L2 0.0029609480407088995). CPU
Inductor versus TLM differed in just 1/61,440 values (relative L2
0.00002958305412903428). This supports a compiled-versus-eager numerical-path
explanation of most of the isolated norm discrepancy; it is not a board
measurement or a full-model acceptance result.

A [non-default precision-cast experiment](vl_reason_norm_precision_experiment.tgz)
set Inductor `emulate_precision_casts=True` for this diagnostic only. Both
CPU Inductor and standalone DP TLM then matched the eager CPU norm bitwise
for all 61,440 values. The setting was not applied to the board or product;
two-layer and 113-point validation are required before considering any change.
A two-layer TLM experiment using the same non-default setting is in progress.

A [full-checkpoint Qwen3.5 CPU diagnostic](q35_warmup_reset_cpu.json) compared
`Say hi.` prefill with and without a dummy decode warmup followed by cache
reset. The resulting logits, all 64 captured attention/linear-attention cache
tensors, and recorded cache metadata were bitwise equal; both first token IDs
were 12675. This validates the reset on the CPU path only, not the queued
board implementation.

The [two-layer precision-cast experiment](vl_reason_pair_precision_tlm.tgz)
repeated the reason-prompt DP TLM pair with non-default
`emulate_precision_casts=True`. Its diagnostic reported pass against available
floors; layer-2 value-cache relative L2 was 0.0 in both prefill and decode,
versus 0.010808053152941194 and 0.010873686087227104 in the default run. The
CPU-Inductor floor again failed to compile due to `vf.cvt.e4m3.f64` and is
missing. Layer-3 value-cache relative L2 remained 0.025762162718081426 in
prefill and 0.025658220925427053 in decode, within the recorded available
floor rule. This experimental setting was not applied to the board or product;
no complete-model or original-audit pass is claimed.

A [full-checkpoint VL CPU multi-turn diagnostic](vl_warmup_multiturn_cpu.json)
compared the registered Alice conversation before and after dummy decode
warmup and reset. Both turns' generated text and token IDs were identical
between the two CPU runs. This does not establish a board multi-turn result.

A [full-checkpoint VL CPU image diagnostic](vl_warmup_image_cpu.json) ran the
registered shapes image after a factual prompt, with and without the earlier
dummy decode warmup. Both CPU runs generated identical token IDs and answered
`circle, square, triangle`. This is not a board image result.

In addition to the original single-token-tail DP TLM replay, a separate
full-checkpoint DP TLM replay of the grouped-tail path is running. Both use
the same registered reason prompt and compare first-step logits against the
saved CPU paths; neither has produced full-model logits yet.

The [full-checkpoint Qwen3.5 CPU quality run after warmup](q35_warmup_quality_cpu.tgz)
passed all five registered applicable prompts, including multi-turn, against
the archived CPU reference with the existing scoring rule. This is a CPU
result; Qwen3.5 board quality and no-host evidence remain pending.

A [full-model CPU fused-normalization approximation](vl_reason_cpu_fusednorm_prototype.tgz)
tested the interaction of normalization and attention numerics. With math
SDPA, whole, legacy and grouped paths all gave `We` and `Bob` equal logits of
38.0; the argmax selected `We`. Grouped-path logits differed from the original
CPU flash reference by relative L2 0.04327630251646042. With CPU flash SDPA,
the grouped approximation selected `Bob` with logits 41.0 versus `We` 37.5.
This is an approximate CPU diagnostic, not the full TLM or board logits;
it illustrates sensitivity of the leading tokens and does not change any gate.

A third DP TLM attempt is running the same 30-token reason prompt as one
whole prefill, to compare its first logits directly with the whole-prompt CPU
reference. Its prefill segmentation differs from the board's chunked path and
will be reported separately; it has not produced logits yet.

A [selected-output DP TLM LM-head diagnostic](vl_reason_lmhead_dp_tlm.tgz)
used CPU-captured final hidden states from whole, legacy-chunked and
grouped-tail reason-prefill paths, and the union of their CPU top-five output
rows (`To`, `We`, `From`, `Let`, `Bob`). All five TLM output logits matched
the corresponding CPU logits bitwise for each of the three hidden states.
These are selected output rows with CPU-captured hidden states, not a
full-model TLM or board logit measurement.

The [five-prompt board snapshot](vl_nohost_v5_five_prompts.json) adds the
registered multi-turn test as passed: the first response acknowledged Alice
and the follow-up answered `Alice.`. Both turns and the previous four prompts
recorded zero host reads in every decode step. The image prompt, persistent
session and scripted chat are still pending. This is not full VL acceptance.


The [complete warmup-retry board quality result](vl_nohost_v5_quality.json)
passed all six registered VL prompts against the archived CPU answers. The
image answer was `circle, square, triangle` with all six token IDs matching
the reference. Across seven turns, all 77 executed decode steps recorded
zero host reads and zero new graphs. The reported 822 parameter tensors and
four buffers were on TNCA; the captured prefill, decode, selection and vision
wrappers reported no CPU numerical fallbacks. This result does not change the
failed previous attempts. VL persistent-session and scripted-chat checks are
still pending, so this is not complete VL acceptance.

For the image turn on the 100 MHz run-2 board, model prefill was
0.18079277384592407 input tokens/s, excluding the separately recorded
182.17397727200296 s vision call. Decode executed 11 steps at
0.010688780698798469 steps/s, with 6,215 launches and 25,692,997,580
device cycles. The six returned token IDs divided by generation wall time
(excluding output retrieval) give 0.003972640770455757 IDs/s. The estimated
decode non-device wall residual was 0.7503392576114729; it includes host
work, transfers, waits and possible counter overlap, so it is not host-only
time. Decode executed all 11 steps even when EOS was reached earlier.
The vision setup recorded host reads during compilation; these are not
decode reads.

A [CPU-only synthetic session diagnostic](event_scan_cpu.json), reproducible
with `tools/models/diagnose_event_scans.py`, measured existing runtime event
history scans; it did not access a board or model. The current session keeps
terminal events and scans that history at submission, drain and module unload.
With 1,000 and 60,000 terminal events respectively, `unfinished` took
0.00009780552238225937 and 0.00646544074639678 seconds per scan on this CPU.
The board quality record also shows decode wall time growing while device
cycles remain similar, so these scans are a plausible contributor, not a
measured board causal attribution or an advertised speedup. No runtime change
was applied to the live board or running TLM attempts.


The [VL warmup-retry board session](vl_nohost_v5_session.json) passed both
the registered two-token teacher check and persistent-session repeat on the
100 MHz run-2 image. The first conversation generated `Hello! How can`;
after an intervening conversation, the repeat had no bitwise logit mismatch
and introduced zero new graphs. Each of the three conversations had zero
host reads on all three decode steps and no compiled-wrapper CPU numerical
fallback. The first prefill recorded 84 compiler constant-folding reads,
separately from decode. Its model prefill rate was 0.04642312187807423 input
tokens/s; decode executed three steps at 0.030273089259719835 steps/s with
5,049,325,197 device cycles. Its estimated non-device decode wall residual
was 0.4904710920328885, including transfers, waits and possible counter
overlap; it is not host-only time. The scripted chat remains pending.

The [complete-model whole-prefill DP TLM diagnostic](vl_reason_whole_dp_tlm.tgz)
ran the registered 30-token reason prompt with the same DP profile as run-2.
This whole-prefill segmentation differs from the board's grouped chunks. Its
first-logit top five were Bob 38.5, We 37.75, Let 37.5, To 36.5 and From
33.75, with top-two gap 0.75. The matching whole-prefill CPU reference gave
Bob 38.25, We 37.25, To 37.25, Let 36.75 and From 33.75, with gap 1.0.
TLM versus that CPU reference had relative L2 0.03805557265877724 and maximum
absolute difference 1.4375, while the top token was the same. Thus there is a
measured numerical difference, not merely a flipped near-tie, and this
whole-prefill TLM run did not reproduce the earlier board failure. The grouped
and legacy chunked full-model TLM runs remain in progress. This is uncalibrated
SystemC evidence, not a board throughput or timing result; the prior failed
board attempt remains unchanged.

A separate [event-scan candidate patch](event_active_candidate.patch) retains
historical events for dependency identity and keeps a bounded index of active
events for submission, drain and code eviction. It has **not** been applied to
the live board queue or released code. The [candidate evidence](event_active_candidate.tgz)
records 113/113 unchanged in the compile-only audit, 5/5 standard Torch
batch tests on DP TLM, 21 synthetic TLM submissions including an old-event
dependency, and 5/5 graph-replay/recovery tests on an isolated RTL simulator.
The first audit start rejected a mistyped tree hash; the next reported 113 not
compared because the staged ISA was missing, and a subsequent incomplete run
was stopped after a missing tool dependency was found. Those attempts remain
in the archive. A TLM attempt to run the replay test was refused because that
test requires isolated RTL; the later RTL run is reported separately. There is
no board A/B measurement or speedup claim for the candidate.

The [partial VL scripted-chat stdout](vl_script_v5_partial_stdout.log)
records the first board turn answering `Paris`; the second turn had started.
This partial output is not a completed transcript. Its prefill call took
477.3567802280013 s and the 31 executed decode steps averaged
0.020542654856751778 steps/s; these steps can include work after EOS. The
per-step transfer evidence for the script is pending its final transcript.

The [complete-model grouped-prefill DP TLM replay](vl_reason_grouped_dp_tlm.tgz)
used the same reason prompt and 8+8+8+6 segmentation as the corrected board
path. Its top five were Bob 38.5, We 37.75, Let 37.5, To 36.5 and From
33.75 (top-two gap 0.75). The same-segmentation CPU reference gave Bob 38.25,
We 38.0, To 37.25, Let 36.75 and From 34.0 (gap 0.25). TLM versus that CPU
reference had relative L2 0.04239596053957939, maximum absolute difference
1.5 and the same argmax. All 151,936 first-step TLM logits were bitwise equal
to the independently run whole-prefill DP TLM logits. This is uncalibrated
model evidence, not a board logit measurement. The older single-token-tail
full-model DP TLM replay is still pending and the prior board failure remains
recorded.

The [two-turn VL scripted-chat stdout](vl_script_v5_partial_two_stdout.log)
adds the follow-up answer `Rome`; the script then reset the session and began
`Say hi.`. The follow-up prefill call took 455.7652571790095 s, and its 31
executed decode steps ran at 0.014760389863401634 steps/s; this includes any
work after EOS. This remains a partial transcript; per-step transfer evidence
is pending completion.

The [completed VL script transcript](vl_nohost_v5_script.json) records three
passing turns, `Paris`, `Rome` and `Hello! How can I help you today? 😊`.
All 93 executed decode steps had zero host reads and zero new graphs; all
822 parameters and four buffers were on TNCA, and the captured prefill,
decode and selection wrappers reported no numerical host fallback. The three
prefill rates were 0.04189738331661979, 0.07679391846722787 and
0.06509714113399628 input tokens/s. The 31 executed decode steps in each turn
ran at 0.020542654856751778, 0.014760389863401634 and
0.011694609412099382 steps/s respectively. Decode may continue after EOS.
Together with the previously recorded quality and session checks this
completes the registered VL stage; it does not establish Qwen3.5 success.

The [Qwen3.5 v5 failed attempt](q35_nohost_v5_failed.tgz) began at 12:50 UTC
and stopped at 13:00 UTC while compiling the dummy decode against an empty
cache. Inductor produced a three-dimensional reduction that the TNCA compiler
could not lower (`ValueError: too many values to unpack`); the quality stage
did not run. The retry prepopulates the real prompt cache, separately records
prefill and decode warmup, clears the cache, and then executes the unchanged
zero-host-read checks. The source and output are separate from the v5 archive;
the retry's result is pending.

The [full-checkpoint CPU comparison](q35_populated_warmup_cpu.json) of the
populated-cache warmup found identical prompt logits and identical captured
cache state after reset, versus a fresh session. This is a CPU diagnostic,
not a board result.

The active-event index is now applied to the development branch. A matched
standard Torch FP32 TLM comparison, recorded in
[speed diagnostics](../speed/matched_tlm_20261031.tgz), ran 1,000 matmuls
after 1,027 previous events: host wall time was 3.190009760670364 s before
and 2.8428647201508284 s after. Both were bitwise equal to the reference
(0/256 mismatches); the final call had 0.265869140625 instructions/MAC,
3553 device cycles, one launch, and zero code loads and evictions in both.
This change does not reduce launch or reload counts. Its separate 113/113
default-code audit and TLM/RTL checks are in `event_active_candidate.tgz`;
board A/B remains pending and the fixed board source is unchanged.

The subsequently completed [legacy single-token-tail DP TLM replay](vl_reason_legacy_dp_tlm.tgz)
reproduced the old wrong first token: its top five were We 38.5, Let 37.0,
Bob 37.0, To 36.75 and From 33.75, with gap 1.5. The CPU run with the
same 8+8+8+1+1+1+1+1+1 segmentation also chose We (38.25; Let 37.0,
To 36.5, Bob 36.5, From 34.0; gap 1.25). TLM versus that matching CPU
segmentation had relative L2 0.04330403357744217, max absolute difference
1.40625 and equal argmax. Against the grouped CPU reference it had relative
L2 0.053693052381277084 and a different argmax. This localises the old
failure to a segmentation-dependent path, with a separate measured CPU/TLM
numeric difference; it does not identify the precise internal operation.
The corrected grouped board quality result remains separate.

The [Qwen3.5 populated-cache warmup board session](q35_nohost_v6_session.json)
passed its two-token teacher check and persistent-session repeat. The three
calls returned `Hi there! How`, `Understood. I`, and `Hi there! How`; all
nine executed decode steps recorded zero host reads and zero new graphs.
All 924 parameters and four buffers were on TNCA and the captured prefill,
decode and selection wrappers reported no CPU numerical fallback. The first
call measured 0.03319671684739197 prefill tokens/s and 0.01423601763386857
executed decode steps/s, with 11,365,085,627 decode device cycles. The
teacher test's two relative L2 values were 0.06922020028616507 and
0.05113672787073658; they passed the pre-registered CPU-floor-adjusted rule
and matched both reference argmax tokens. The quality stage is still running;
this session alone is not complete Qwen3.5 acceptance.

The [Qwen3.5 quality checkpoint](q35_nohost_v6_partial.json) records the
first of five prompts passing: `Paris`, with all four returned token IDs
equal to the CPU reference. All eleven executed decode steps had zero host
reads and zero new graphs; 924 parameters and four buffers were on TNCA,
and the captured wrappers reported no CPU numerical fallback. The first
prefill ran at 0.03310412431217009 tokens/s and the eleven decode steps at
0.01292878839389162 steps/s, including any work after EOS. Compilation
warmup and post-generation output retrieval are recorded separately. The
other four prompts were not yet recorded at this checkpoint.

The [15:25 UTC quality checkpoint](q35_nohost_v6_two_prompts.json) records
two of five prompts passing. The new arithmetic answer was `42`; its
prefill ran at 0.02921525881795749 input tokens/s and decode at
0.010352622915114129 executed steps/s, including work after EOS. The remaining
three prompts have not yet been scored. The three-hour experiment limit may
expire before all turns finish at the observed rate; this is a risk, not a
recorded timeout.

The [16:13 UTC quality checkpoint](q35_nohost_v6_three_prompts.json) records
three of five prompts passing, adding the reasoning answer `Bob`. Its prefill
rate was 0.02334469452375064 input tokens/s and decode rate was
0.008263454449353351 executed steps/s, including work after EOS. The quality
run continues; the remaining two prompts are not yet scored.

The [final checkpoint of the original Qwen3.5 quality attempt](q35_nohost_v6_four_prompts_timeout.json)
contains four passing prompts; the added instruction-following answer was
`OK`. Its 11 executed decode steps had zero host reads, zero new graphs and
no captured CPU numerical fallback; all 924 parameters and four buffers were
on TNCA. Prefill was 0.022209985250415985 input tokens/s and decode was
0.007195171218281826 executed steps/s, including work after EOS.

The experiment reached its registered 10800-second limit and terminated the
quality process; its [report](q35_nohost_v6_timeout_report.json) records the
step as `timeout`, and the [original raw attempt](q35_nohost_v6_timeout.tgz)
is preserved. The checkpoint's `running` status reflects the forced process
termination, not completion. At 16:51:52 UTC a separate registered reduced
proof for the missing `multiturn` prompt started on the same run-2 image;
it leaves the original timeout unchanged. At that checkpoint the speed A/B stages
were queued behind the segmented coverage check.


The separate [multi-turn board attempt](q35_multiturn_segment_failed_20261031.tgz)
finished at 18:43 UTC and **failed** the registered quality rule. Its first
answer, `I have remembered your name, Alice.`, differed from the CPU reference
`Hello Alice, I have noted your name.` from the first token and neither
normalized answer contained the other. The follow-up `Alice` matched all four
reference tokens. This result does not convert the original timeout or the
failed prompt into a pass. The [summary](q35_multiturn_segment_failed_20261031.json)
records both turns: 11 decode steps each, zero decode host reads, no captured
CPU numerical fallback, all 924 parameters and four buffers on TNCA. Prefill
rates were 0.0305566080 and 0.0337049922 input tokens/s; decode rates were
0.0126348384 and 0.0090442224 executed steps/s. CPU and DP TLM first-logit
comparison is in progress. After preserving this failure, a new safety-checked
queue started the previously gated acceleration A/B sessions at 18:49 UTC on
the already loaded image; these sessions do not change the quality judgment.


The [full-checkpoint CPU first-logit replay](q35_multiturn_cpu_logits_20261031.tgz)
used the same 27-token first turn. Whole-prompt CPU top five were `Hello`
22.25, `I` 21.75, `Alice` 19.375, `Nice` 19.375 and `Your` 19.0; gap 0.5.
With the board's 8+8+8+3 grouped chunks they were `Hello` 22.0, `I` 21.875,
`Alice` 19.375, `Nice` 19.125 and `Your` 18.75; gap 0.125. CPU grouped vs
whole relative L2 was 0.0589486696, maximum absolute difference 0.578125,
and both chose `Hello`. The board selected `I`, but its full logits were not
captured: the close CPU gap alone does not establish the reason for the board
flip. Matching-source DP TLM is running. No quality judgment is revised.

The [CPU-only attention-backend diagnostic](q35_multiturn_cpu_sdpa_20261031.tgz)
replayed the same full-checkpoint prompt using the PyTorch mathematical SDPA
backend. Its grouped 8+8+8+3 path selected `Hello` (22.0) over `I` (21.875),
a gap of 0.125; whole prefill selected `Hello` (22.125) over `I` (21.875),
a gap of 0.25. Compared to the corresponding default CPU attention path, the
grouped and whole logits had relative L2 differences 0.0642956123 and
0.0664141327, with the same argmax in both cases. This isolates a CPU backend
effect but neither identifies the board's selected-token cause nor captures its
logits. An auxiliary CPU norm override in the same archive failed with an
attribute error (`Qwen3_5RMSNorm` uses `eps`, not `variance_epsilon`); it did
not produce a comparison and is not a pass. The full-model grouped and whole
DP TLM runs remain separate and pending.

Two additional [CPU-only attention approximations](q35_multiturn_cpu_flash_20261031.tgz)
also chose `Hello` on the grouped prompt. The unnormalised-probability rounding
prototype produced `Hello` 22.25 versus `I` 21.75 (gap 0.5), and its SVE-exp
prototype produced 22.25 versus 21.625 (gap 0.625). Against the default CPU
grouped logits, their relative L2 differences were 0.0612189882 and
0.0636935830; both had the same argmax. These are CPU prototypes, not a
bit-exact model of the board attention, and do not establish the board's
cause of the first-token difference.

The [full-checkpoint CPU layer comparison](q35_multiturn_cpu_layers_20261031.tgz)
compared whole prefill against the board-style 8+8+8+3 grouping for the same
27-token prompt. Decoder layers 0–3 were bitwise identical across all 69,120
values per layer. Layer 4 first differed at token position 13 (zero based),
with 2/69,120 mismatches and maximum absolute difference 0.00048828125; its
last token remained identical. At layer 7, the last-token relative L2 was
0.0427951515; at layer 31 it was 0.0615154132. This isolates a CPU
segmentation effect but does not determine the source of the board's
first-token difference. Matching-source full-model DP TLM remains pending.

The [CPU layer-4 component probes](q35_multiturn_cpu_layer4_20261031.tgz)
showed identical whole/grouped input normalization and QKV, Z, A and B
projections. The linear-attention gated-normalization input from the attention
core first differed at flattened head vector 283, with 22/110592 unequal bit
patterns and max absolute difference 1.1920928955e-7. The other normalization
input Z was bitwise identical. After normalization, 20/110592 patterns differed;
the attention output projection differed in 14/69120 values. These are CPU
segmentation comparisons and do not establish the board failure's cause. The
first input-probe setup failed locally before sending its source; the resulting
empty remote program exited without testing anything and is not a pass. The
separate v2 input-probe result is the one reported here.

The [CPU layer-4 convolution and gated-delta probes](q35_multiturn_cpu_conv_delta_20261031.tgz)
found bitwise identical causal-convolution outputs and Q, K, V, g and beta
inputs for whole versus grouped prefill. The chunked gated-delta output first
differed at token position 8, with 22/110592 bit mismatches and maximum
absolute difference 1.1920928955e-7. The whole and grouped calls have different
chunk boundaries and state inputs by construction. This localizes the earliest
observed CPU segmentation difference to that computation; it is not a board or
DP TLM measurement and does not establish the cause of the board's first token.

A separate [preregistered whole-prefill multi-turn board diagnostic](q35_whole_board_queue_20261031.tgz)
waits behind the existing speed and hoist queue without programming an image.
It keeps the original prompt and CPU scoring rules but uses `--prefill-chunk 0`
instead of 8, so it is a distinct reduced proof, not a revision of the failed
registered grouped-prefill result or original timeout. Its first queue setup
failed before touching the card because the recovery-helper path was wrong;
the corrected v2 queue is waiting. Before executing it checks the source and
manifest hashes, takes the existing queue lock, and verifies the loaded UUID,
clock, image-file hash and card idle state. Its result is pending.

The [22:30 UTC whole-prefill first-turn checkpoint](q35_whole_turn0_checkpoint_20261031.log.gz)
prints `I have remembered your name, Alice.`, identical to the first answer
of the failed grouped attempt. The unchanged CPU reference is `Hello Alice,
I have noted your name.`; the first turn therefore cannot satisfy the registered
containment rule. This is a live stdout checkpoint, not the final structured
quality report: the second turn and its counters were not yet recorded at the
checkpoint. The original grouped failure and timeout remain unchanged.

The [whole-prefill full-checkpoint DP TLM logits](q35_multiturn_tlm_whole_20261031.tgz)
use the same 27 input tokens and model revision as the CPU comparison. TLM
chose `I` (22.25) over `Hello` (22.0), gap 0.25. CPU whole-prefill chose
`Hello` (22.25) over `I` (21.75), gap 0.5. Across 248,320 logits, relative
L2 was 0.0774650127, maximum absolute difference 1.0, and 242,662 bit patterns
differed. This is a genuine logits difference rather than evidence only of an
exact tie. The board emitted the same first token, but board logits were not
captured, so this does not prove the complete device-side cause. The grouped
DP TLM run remains pending.

The [instrumented hidden-state probe](q35_multiturn_layers_pending_20261031.tgz)
is running separately on CPU and DP TLM with the same whole-prefill prompt,
model and source snapshot. It requests hidden states as outputs, which can
change the compiled graph, and its CPU/TLM comparison is diagnostic only.
The CPU run completed and its first logits matched the original CPU reference
bitwise (0/248,320 mismatches). The DP TLM run and automated comparison remain
pending; the comparison also checks whether instrumented TLM logits match the
uninstrumented TLM result.

The [completed whole-prefill board diagnostic](q35_whole_board_complete_20261031.json)
and [raw archive](q35_whole_board_complete_20261031.tgz) report failed chat
quality: first-turn containment failed and the second `Alice` turn matched all
four reference tokens. Across both turns the 22 steady decode steps had zero
host reads, all 924 parameters and four buffers were TNCA, and no captured
CPU numerical fallback. Prefill was 0.0567254275 and 0.0822505149 input
tokens/s; decode was 0.0142777756 and 0.0127684518 executed steps/s. These
include work after EOS. Prefill and warmup did record host transfers: the zero
read claim is only for the 22 steady decode steps. This reduced diagnostic
does not replace the grouped failure or original quality timeout.

A separate [two-layer diagnostic](q35_multiturn_first_layers_pending_20261031.tgz)
was started to obtain early embedding/first-layer comparisons before the full
32-layer TLM probe. It truncates the checkpoint and requests hidden outputs, so
it is not a full-model or board acceptance result. The comparison checks the
partial CPU embedding and first-layer output against the full CPU capture. Its
CPU and DP TLM results are pending.

The [two-layer headless diagnostic](q35_multiturn_first_headless_pending_20261031.tgz)
was also started to avoid simulating the large vocabulary projection while
localizing the first-layer output. It truncates the model, replaces `lm_head`
with identity, and requests hidden outputs, so it is diagnostic only. The
comparator checks the CPU embedding and first layer against the full CPU
capture; results are pending.

The first headless CPU diagnostic [failed while saving](q35_first_headless_v1_failed_20261031.log.gz):
the identity head shared storage with the final hidden tensor, and safetensors
refused duplicate storage. Its paired TLM run was stopped because it would hit
the same save error; no numerical result is claimed. A separately recorded
[v2 diagnostic](q35_first_headless_v2_pending_20261031.tgz) clones outputs
before saving and is running alongside the original full-head diagnostic.

The [first-layer FP8 dot probe](q35_first_fp8_dp_20261031.tgz) captures the
real CPU activation at position 26 of the same prompt, and compares the
ISA-defined four-product `m.mma.dp` accumulation emulation with the registered
CPU FP8 reference, both using IEEE activation division. QKV, Z, gate and up
projections had respectively 0/8192, 0/4096, 0/9216 and 0/9216 bit mismatches;
the captured module outputs also matched their CPU reference. Against a
separate exact BF16 oracle, both the grouped emulation and registered reference
differed in one gate value (max 3.8146973e-6) and one up value (max
4.7683716e-7). This CPU arithmetic probe rules out a grouped-vs-reference
difference only for these sampled inputs; it is not TLM, board, or a proof
about other positions and layers.

The [full-model CPU scale/FMA experiments](q35_cpu_scale_fma_pending_20261031.tgz)
compare multiplying the activation maximum by rounded 1/448, FP32 multiply-add
contraction, and their combination, against the registered IEEE-division,
non-contracted reference. These variants are CPU approximations and are not
claimed to reproduce every TNCA reciprocal or board operation. Results are
pending.

The [completed CPU scale/FMA comparisons](q35_cpu_scale_fma_20261031.tgz)
show that whole-prefill FMA with IEEE scaling produced a tie between `I` and
`Hello` at 21.75; `argmax` selected the lower token ID `I`. Its logits had
relative L2 0.0686228648 and max absolute difference 0.8125 against the
registered CPU whole-prefill reference, and relative L2 0.0738517269 against
the whole-prefill DP TLM logits. Thus a small arithmetic change can change the
selected token, but FMA alone does not reproduce the TLM logits or establish
the board cause. The same FMA experiment with grouped prefill selected `Hello`
(22.25 vs `I` 21.5). Rounded reciprocal alone and reciprocal plus FMA both
selected `Hello` on both prompt paths. These are CPU-only arithmetic
approximations, not a change to the reference or a board-quality pass.

The [completed headless two-layer CPU/DP-TLM probe](q35_first_headless_v2_20261031.tgz)
found 0/69120 embedding bit mismatches. After the first layer it found
63732/69120 differing values, relative L2 0.0119732721 and maximum absolute
difference 0.03125; the partial CPU embedding and first-layer output each
matched the corresponding full CPU capture bitwise. After the second layer's
final norm the relative L2 was 0.0361934826. This is a truncated, instrumented
model; it does not show board intermediates or prove the precise cause.

A [one-layer component probe](q35_first_components_pending_20261031.tgz)
was started to compare normalization, attention projections and MLP outputs.
Its graph hooks and truncation may change compilation; results are pending.

The [isolated first RMSNorm DP TLM probe](q35_first_norm_tlm_20261031.tgz)
used the CPU-captured embedding for all 27 tokens and matched the CPU output
bitwise in 69,120/69,120 elements. It is a standalone graph rather than the
full model; the instrumented full-layer comparison remains necessary. Two
[one-token projection probes](q35_first_projection_tlm_pending_20261031.tgz)
for QKV and MLP gate were started on the CPU-captured activations; results
are pending and will not establish full-prefill behavior by themselves.

The separate [two-layer probe retaining the vocabulary head](q35_first_layers_20261031.tgz)
completed with the same first-layer CPU/TLM mismatch count and relative L2 as
the headless probe. A direct TLM tensor comparison between the two probes found
0/69120 bit mismatches each for the embedding, first-layer output and second
layer final-norm output. Both remain truncated diagnostic graphs, not board
intermediates or full-checkpoint acceptance.

The [single-layer component comparison](q35_first_components_20261031.tgz)
found CPU and DP TLM bitwise identical at embedding, input RMSNorm and all
QKV, Z, A and B input projections across 27 tokens. The gated-normalization
output differed in 67,305/110,592 values (relative L2 0.0029556095); the
linear-attention output differed in 60,841/69,120 (relative L2 0.0054668272).
The CPU instrumented first-layer output was bitwise equal to the full CPU
capture, and the instrumented TLM first-layer output was bitwise equal to the
separate two-layer TLM. The probe does not yet distinguish gated-delta output
from gated normalization. The standalone [one-token QKV and gate DP TLM](q35_first_projection_tlm_20261031.tgz)
matched their CPU-captured outputs 8192/8192 and 9216/9216 bitwise; their
shape and graph differ from the 27-token prefill.

A [deeper convolution/delta probe](q35_first_delta_pending_20261031.tgz)
was started to capture convolution output, gated-delta inputs and output, and
the two gated-normalization inputs. Its extra hooks may affect the compiled
graph, so it also checks the instrumented whole first-layer output against the
previous TLM result.

The [standalone gated RMSNorm DP TLM probe](q35_gate_norm_tlm_20261031.tgz)
used the same CPU-captured attention-core and gate operands on both routes. It
differed in 38067/110592 output bits, relative L2 0.0027909812, max absolute
difference 0.015625. Unlike the end-to-end comparison, this demonstrates a
difference within the isolated gated-norm operation on identical inputs. It
is still not a board measurement or proof that the operation is the only
source of the quality failure. An [instrumented decomposition](q35_gate_parts_tlm_pending_20261031.tgz)
is running to compare variance, rsqrt, normalized and weighted values, SiLU,
and final product; it checks its CPU reconstruction and compares its final
TLM output with the uninstrumented standalone operation.

The [completed gated-norm decomposition](q35_gate_parts_tlm_20261031.tgz)
reconstructed the captured CPU output bitwise and reproduced the uninstrumented
standalone TLM output bitwise. CPU/TLM variance differed in 427/864 FP32 values
(relative L2 6.17e-8), rsqrt in 377/864 (4.72e-8), and the normalized FP32
values in 41461/110592 (6.46e-8). After BF16 weighting, 28575/110592 values
differed (relative L2 0.0027097). SiLU also differed in 40209/110592 FP32
values, with relative L2 4.57e-8. Final output difference remained
38067/110592 (relative L2 0.00279098). These results identify multiple small
operator differences and their BF16 impact in this isolated test; they do not
yet identify the specific implementation defect or prove full-model causality.

The deeper [convolution/delta comparison](q35_first_delta_20261031.tgz)
found 56254/221184 differing convolution output values (relative L2 0.0025008),
while QKV inputs were bitwise identical. However, adding these hooks changed
the TLM first-layer output versus the previous two-layer probe in 62739/69120
values (relative L2 0.0097398), so this is an instrumentation-dependent clue,
not proof of the original graph's earliest divergence.

The [rounding reconstruction and first precision-attempt failures](q35_gate_rounding_20261031.tgz)
show the generated Inductor kernel converted the normalized BF16 intermediate
to FP32 without its required round. Recomputing the captured TLM intermediate
with that cast omitted reproduces 110592/110592 weighted outputs; honoring the
BF16 cast gives 28575 mismatches. The first precision-wrapper attempts failed
before a result because Triton was imported before the tree path was set. A
[separately recorded retry](q35_precision_cast_retry_20261101.tgz) fixes import
order, declares non-default `emulate_precision_casts=True`, and is running
further checks. Its standalone gated-norm result has 1/110592 bit mismatch
(relative L2 1.3869843e-6, max 3.0517578e-5), down from 38067; it is not
bitwise exact, so this is not a passed exact gate or a board result.

The [precision-cast gated-norm TLM and decomposition](q35_gate_precise_v2_20261101.tgz)
reduced the standalone output mismatch from 38067 to 1/110592, max absolute
difference 3.0517578e-5. The instrumented precision-cast final TLM output
matched its uninstrumented counterpart bitwise. The remaining mismatch was
at flattened head row 755, element 43: CPU versus TLM variance was
3.1553292501484975e-5 versus 3.155328886350617e-5; their rsqrt values were
175.26795959472656 versus 175.26797485351562. The normalized value crossed a
BF16 rounding boundary; SiLU was identical at that coordinate. This is one
recorded case, not a general exactness or board pass.

The [precision-cast two-layer TLM](q35_first_headless_precise_v2_20261101.tgz)
reduced first-layer mismatches from 63732 to 1887/69120 and relative L2 from
0.0119732721 to 0.0004243143; after the second layer final norm, mismatches
were 51894/69120 and relative L2 0.0156156691. Neither is bitwise exact.

The [compensated-reduction research](q35_compensated_research_20261101.tgz)
retains a failed first full-kernel attempt at a two-million-cycle TLM deadline
and the separately registered higher-deadline retry. On 864 real activation
rows, compensated FP32 reduction matched high-precision variance 864/864,
versus 427/864 for the ordinary reduction; cycles were 1497554 versus 179093,
with one launch each and no loads during launch. On real gated-norm inputs,
an explicit-BF16-cast standalone Triton kernel matched 110592/110592 CPU
outputs both with ordinary reduction (749101 cycles, 438912 instructions) and
compensation (2568426 cycles, 1856736 instructions). A random finite-input
case also matched 16384/16384 in both paths (110177 and 380483 cycles). These
are standalone kernels with a different reduction schedule than the original
Inductor graph, not proof of general exactness, no board claim, and not a
merged compiler fix.

The [precision-cast block-size comparison](q35_gate_blocks_precision_20261101.tgz)
used the same real gated-norm inputs with non-default `TNCA_MAX_BLOCK` 32, 64
and 128: each matched 110592/110592 outputs. A separate random finite-input
probe for block 32, spanning four scales, had 1/65536 mismatch (max absolute
0.0009765625). The candidate therefore does not pass a bitwise exactness gate.
The [two-layer and full-model b32 TLM runs](q35_precise_b32_model_pending_20261101.tgz)
are running with automatic comparisons; they are diagnostics, not board runs.

The [random gated-norm follow-up](q35_gate_random_ieee_20261101.tgz) found
that its only b32 precision-cast mismatch had bitwise-identical CPU/TLM
variance, rsqrt, normalized and BF16-weighted values at the failing element.
SiLU differed by one FP32 ULP (0.5392377377 versus 0.5392376781), causing
the final BF16 tie to round differently. Both ordinary and compensated
standalone means retained this one mismatch. Adding declared non-default
`TNCA_IEEE_DIV=1` to precision casts and block 32 gave 0/65536 mismatches on
this same random case. This is still finite single-shape evidence, not a
general exactness or board pass. [Two-layer, full-model, component and
convolution TLM comparisons](q35_precise_b32_ieee_model_pending_20261101.tgz)
with those knobs are pending.

The [IEEE-division research](q35_div_signedzero_research_20261101.tgz)
first confirmed `+0 / 2` produced `-0`; a [signed-zero-only patch](q35_div_signedzero_research_20261101.patch.gz)
fixed 16/256 boundary mismatches and made the captured standalone convolution
221184/221184 exact. However, both the old and fixed path retained 490/65536
differences on finite random inputs, including extreme denominators and
subnormals. That patch was not merged.

The later [scaled/residual candidate and failed attempts](q35_div_v7_research_20261101.tgz)
and [candidate patch](q35_div_v7_research_20261101.patch.gz) improve the
ordinary IEEE-division path. Nine random sets total 589824/589824 bitwise
matching finite inputs, 2928/2928 signed boundary comparisons match (NaNs
checked by classification), and 32768/32768 near-overflow values match. In
one 65536-input TLM test the candidate used 870797 device cycles and 634880
instructions, versus the baseline's 526831 and 155648, one launch and no
code loads during launch in each. On current integration source the existing
explicit `tl.div_rn` path matched 65536/65536 before and after, but cycles
increased 741451 to 1036176 and instructions 479232 to 892928. The
[current-source default-code audit](q35_div_v7_current_audit_20261101.tgz)
compared 113/113 unchanged. Two-layer Qwen3.5 compilation with the candidate
failed from vector-register exhaustion, including separate attempts with
window reuse and 16-wide blocks. It is not merged or board-qualified.

The [explicit-division wrapper experiment](q35_div_wrapper_research_20261101.tgz)
reused the existing `tl.div_rn` lowering for non-default IEEE division; its
[candidate patch](q35_div_wrapper_research_20261101.patch.gz) does not alter
the default branch. On DP TLM it matched 65536/65536 finite random inputs,
but failed 4/2928 signed boundary cases and 128/32768 near-overflow cases:
a finite maximum result became infinity. The random run used 741451 device
cycles and 479232 instructions, with one launch and no code loads during it.
The candidate fails the exactness gate and is not merged or board-qualified;
no 113-point audit or model speed claim is inferred from this failed probe.

The completed [full-depth instrumented CPU/TLM comparison](q35_all_layers_tlm_compare_20261101.tgz)
shows matching embedding bits, then 63732/69120 mismatches (relative L2
0.0119732721) after layer 1 and 67141/69120 (relative L2 0.0888381749)
after layer 32. Instrumented TLM logits differ from instrumented CPU logits
by relative L2 0.0765310973. Critically, the instrumented TLM logits also
differ from the original uninstrumented whole-prefill TLM logits by relative
L2 0.0701584443 (242415/248320 differing bits), whereas instrumented and
original CPU logits match bitwise. Instrumentation therefore changes the TLM
graph substantially; these layer comparisons suggest where to investigate
but do not establish where the original board path first diverges. No board
logits or correctness pass are inferred.

The completed [full-checkpoint grouped-prefill TLM comparison](q35_multiturn_tlm_grouped_20261101.tgz)
uses the exact Qwen3.5 multi-turn prompt and input tokens. On grouped chunk 8,
TLM chose `I` (22.125) over `Hello` (22.0), a 0.125 gap; the matching CPU
chunked route chose `Hello` (22.0) over `I` (21.875), also a 0.125 gap. Their
248320 logits have relative L2 0.0752601773, maximum absolute difference
1.25 and 242648 differing bits. The first-token flip occurs near a tie but is
accompanied by broad numerical differences. Board logits were not captured,
so this does not establish the complete cause of the board failure.

A [pending full-checkpoint functional TLM diagnostic](q35_untimed_whole_pending_20261101.tgz)
uses the same prompt with timeline memory in untimed mode, in three parallel
runs: ordinary baseline, non-default b32 precision casts, and b32 precision
casts plus `TNCA_IEEE_DIV=1`. The prior timed runs remain active. The attached
comparison service first compares untimed baseline against timed baseline,
then reports each route's logits versus CPU. This experiment measures no
cycles, has not completed, and cannot establish board quality or an accepted
fix; the timing/memory-model change must be checked for numerical equivalence.

The completed [full-checkpoint precision-cast, b32 and IEEE-division TLM
comparison](q35_full_precise_b32_ieee_tlm_20261101.tgz) selects `I` (22.125)
over `Hello` (21.875), while the same whole-prefill CPU reference selects
`Hello`. Relative L2 to CPU is 0.0729162544, versus 0.0774650127 for the
ordinary whole-prefill TLM; 242712/248320 candidate logits differ bitwise.
The candidate therefore does not resolve this first-token quality failure and
is not a board pass. The other precision and untimed diagnostics remain pending.

The completed [full-checkpoint precision-cast and b32 TLM diagnostic](q35_full_precise_b32_tlm_20261101.tgz)
with default division selects CPU's first token `Hello` (22.25 versus `I`
22.0), unlike ordinary whole-prefill TLM. It still differs from CPU in
242077/248320 logits, with relative L2 0.0671156868 and max absolute error
1.0078125. This is a promising first-token observation, not bitwise equality,
complete generated-answer quality, a proven board fix, or a board result.
The setting uses an Inductor precision-cast wrapper and non-default block size;
it is not merged.

The separate whole-prefill precision-cast/b32 board diagnostic was registered
with unchanged multi-turn scoring and the already-loaded image; it is a reduced
proof and does not change the original failures. Its [first queue start](q35_precision_b32_queue_initial_failure_20261101.log)
failed before registration because a recovery module path was wrong. The
[first registered stage](q35_precision_b32_board_launch_failure_20261101.tgz)
failed before model execution: the wrapper imported an older installed TNCA
package under isolated Python. A revised wrapper explicitly selects the new
independent runtime copy before importing Triton; a non-board import check
confirmed that `tnca` resolves to that copy. The separately registered retry
started its stage at 2026-11-01 03:59:04 UTC after the shared lock, card idle,
image UUID/hash, clock and source-hash checks. Its registration SHA-256 is
`7d5edef5c6d94734970835b552db7c7abff85f82759f4126808081606fa2b0bb`.
The [queue](queue_q35_precision_b32_retry_20261101.py) uses a 600-second
poll interval and never programs an image. No result is yet available.
