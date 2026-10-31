# Board campaign progress, 2026-10-31 01:28 UTC

The author explicitly authorized final-image loading without further approval,
including the 433 MHz exception. The 15-minute idle rule and protection of
other users remain in force. The corrected image is loaded; its identity,
measured DATA clock, and completed identity/HBM checks are recorded in
[`README.md`](README.md). Its routed HBM setup violations and conditional CDC
limitations remain recorded in [`../route_prepackage/README.md`](../route_prepackage/README.md).

The active detached board queue is PID 1080780, output
`/home/zzhouch/r41run3/queue_resume_2/queue.jsonl`, with the 113-point stage's
state and raw records under
`/home/zzhouch/r41run3/queue_resume_2/results/original113/r3-s2-original-suite/original/`.
The 113-point stage passed 113/113; its raw records and preliminary performance
comparison are linked from `README.md`. The queue has started the default
513-point memory-bound stage; at the time above both large batches passed
171/171, both medium batches passed 171/171, and both small batches passed
171/171. The default 513-point suite completed 513/513 and the tiled2 suite
had covered 61 configurations with no failures. The queue will run later
registered stages in order, stopping on failure. The
previous identity and HBM clamp failures and fixes are preserved in `README.md`.

Inspect the queue and running processes before any intervention; do not stop
other users' processes or program over an active image. If a stage fails,
retain its output, diagnose it, and resume only with valid image, clock,
software-tree, and prior-result evidence. Do not mark R41 accepted until the
remaining tests and delivery gates actually finish.
Nine original-suite points were slower in the preliminary comparison, including
tiny RoPE by 20.3%; the registered performance recheck remains necessary.
Among the 171 completed large memory-bound points, 12 are over 5% slower and
two strided division points are about 12.4% slower; preserve those comparisons
and distinguish the image change from any new-default optimization decision.
The 171 completed medium points have 13 over 5% slower, including two strided
division points about 12.6% slower; the full point comparison is linked in the
board README and likewise requires registered rechecks.
Across all 513 default points, 39 are over 5% slower and eight over 10% slower
in the descriptive matched comparison; the registered rechecks remain
necessary.

The merged-source strict compiler gate passed 170 unit tests and six IR cases
([`../merge_gate/README.md`](../merge_gate/README.md)); the merged harness
tests passed after correcting a Stage 1 frozen-input test
([`../../../../stage1/frozen_inputs/README.md`](../../../../stage1/frozen_inputs/README.md)).
The Stage 2 integrity checker also passes after the original evidence versions
were preserved; the acceptance judgments remain unchanged and not accepted
([`../../../../acceptance/historical_20261028/README.md`](../../../../acceptance/historical_20261028/README.md)).

The original candidate queue does not include mandatory BF16 Q41-core, XL-core
or final independent reproduction. Qwen3.5-2B checkpoint, independent FP32
reference and pinned official sources are staged and verified; the pilot
passed a two-step CPU check, not a board acceptance run
([`../../../../acceptance/q41_pilot/README.md`](../../../../acceptance/q41_pilot/README.md)).
Board-host PID 1132297 waits for all twelve original candidate queue stages,
verifies their recorded evidence, and then waits for the matching idle card to
run the preregistered one-hour, two-step Q41 pilot. Its log is
`/home/zzhouch/r41run3/q41_pilot_queue/queue.jsonl`. It never loads an image;
it stops if the predecessor or pilot fails. Board-host PID 1133202 waits for
that Q41 pilot to pass, validates its result and then queues a one-hour
`XL41-gemv` pilot on the same idle image; its log is
`/home/zzhouch/r41run3/xl41_pilot_queue/queue.jsonl`. The XL pilot's CPU
preflight and registered inputs are in
[`../../../../acceptance/xl41_pilot/README.md`](../../../../acceptance/xl41_pilot/README.md).
Board-host PID 1134314 waits for that GEMV pilot to pass and then queues the
one-hour, >2 GiB `XL41-stream` pilot; its log is
`/home/zzhouch/r41run3/xl41_stream_queue/queue.jsonl`, and its full-size CPU
preflight is in [`../../../../acceptance/xl41_stream_pilot/README.md`](../../../../acceptance/xl41_stream_pilot/README.md).
Board-host PID 1136414 waits for the stream and Q41 pilots, then queues full
Q41-core, Q41-tail and a second core process with bounds derived from the
measured pilot; its log is `/home/zzhouch/r41run3/q41_full_queue/queue.jsonl`.
Board-host PID 1138206 waits for those sessions, then queues registered XL
transpose, GEMM and projection pilots; its log is
`/home/zzhouch/r41run3/xl41_matrix_queue/queue.jsonl`, with CPU preflight in
[`../../../../acceptance/xl41_matrix_pilots/README.md`](../../../../acceptance/xl41_matrix_pilots/README.md).
Board-host PID 1139463 waits for those matrix pilots, then queues gather,
attention and exploratory recurrence pilots; its log is
`/home/zzhouch/r41run3/xl41_extra_queue/queue.jsonl`. The gather formula does
not include three required boundary rows, so the pilot tests those separately;
the recurrence update equation is unspecified, so that pilot is explicitly
exploratory. Details and CPU records are in
[`../../../../acceptance/xl41_extra_pilots/README.md`](../../../../acceptance/xl41_extra_pilots/README.md).
Full Q41-core, XL-core and clean-archive independent reproduction remain
unmeasured and are not passes. The two specification gaps remain unresolved.
