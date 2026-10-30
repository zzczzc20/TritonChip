# Board campaign progress, 2026-10-30 22:00 UTC

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
171/171 and its first medium batch had covered 78 configurations with 156
successful sample processes and no failures. The queue will run later
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
