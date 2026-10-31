# Restored grid-admission test (pending)

The R9 audit requires rerunning the original opt-in check; the earlier board
result tested the capacity refusal twice and cannot close S2-04.D.4. The
[registered rerun](../../../../../../spec/r41/perf_recheck/grid_admission_rerun.json)
uses the restored `grid_admission_case.py` with the merged candidate software
tree on the restored 100 MHz run-1 image. This tests that specific stack, not
the corrected candidate image or a full-model lifecycle. Its hand-built AOT
kernel is a functional grid-admission diagnostic, not a performance result.

Board-host PID 1143624 waits for all four recheck stages and the hashed run-1
clock verification, then waits for an idle matching card. It does not load an
image. A failure is preserved and stops the waiter; nothing is counted as
passing before the actual board result. The log and output are under
`/home/zzhouch/r41run3/grid_rerun_queue`.

The current physical-grid code takes its base limit from the device. Earlier
run-1 evidence reported a base limit equal to its full advertised capacity,
so the restored opt-in check may fail; that observation is a reason to measure,
not a reason to change the check or assume the rerun's result.
