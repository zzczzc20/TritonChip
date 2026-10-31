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
