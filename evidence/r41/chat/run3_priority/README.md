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
Python stack dumps is queued after the priority chats. It does not change the
model or tolerances and is only a diagnostic until actual results are reviewed.
Board-host PID 1151183 waits for the original chat queue, then requires the same
image, 100 MHz and an idle board. The diagnostic's registration is
[`r3-vl-stall-trace.json`](../../../../spec/r41/chat/diagnostics/r3-vl-stall-trace.json).

The existing chat implementation explicitly transfers visual outputs to the
host and downloads logits for host-side token selection outside the per-model
transfer meter. Its static `cpu_numerical_fallback=false` flag is therefore
not proof of the author's no-fallback and zero-steady-host-read requirements.
No such requirement is marked passing from that flag.
