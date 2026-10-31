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
