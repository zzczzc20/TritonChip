# Restored grid-admission test (pending)

**Descoped by the author on 2026-10-31.** The board queue was cancelled; historical results and registrations below remain unchanged. See [the author directive](../../../../AUTHOR_DIRECTIVE_20261031.md).

The R9 audit requires rerunning the original opt-in check; the earlier board
result tested the capacity refusal twice and cannot close S2-04.D.4. The
[registered rerun](../../../../../../spec/r41/perf_recheck/grid_admission_rerun.json)
uses the restored `grid_admission_case.py` with the merged candidate software
tree on the restored 100 MHz run-1 image. This tests that specific stack, not
the corrected candidate image or a full-model lifecycle. Its hand-built AOT
kernel is a functional grid-admission diagnostic, not a performance result.

Board-host PID 1146076 waits for all four recheck stages and the hashed run-1
clock verification, then waits for an idle matching card. It does not load an
image. A failure is preserved and stops the waiter; nothing is counted as
passing before the actual board result. The log and output are under
`/home/zzhouch/r41run3/grid_rerun_queue_v2`. The earlier unstarted waiter log
is retained in `grid_rerun_queue`.

The current physical-grid code takes its base limit from the device. Earlier
run-1 evidence reported a base limit equal to its full advertised capacity,
so the restored opt-in check may fail; that observation is a reason to measure,
not a reason to change the check or assume the rerun's result.
