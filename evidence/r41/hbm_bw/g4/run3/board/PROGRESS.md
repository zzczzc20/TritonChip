# Board campaign progress, 2026-10-30 18:38 UTC

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
At the time above it had recorded 154 successful sample processes across 77
configurations, with no failures; this is not a completed 113-point suite.
The queue will run later registered stages in order, stopping on failure. The
previous identity and HBM clamp failures and fixes are preserved in `README.md`.

Inspect the queue and running processes before any intervention; do not stop
other users' processes or program over an active image. If a stage fails,
retain its output, diagnose it, and resume only with valid image, clock,
software-tree, and prior-result evidence. Do not mark R41 accepted until the
remaining tests and delivery gates actually finish.
