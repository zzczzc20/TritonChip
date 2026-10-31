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
