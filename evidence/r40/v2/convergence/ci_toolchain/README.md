# CI toolchain version-report failure

Run 38876623180 on source 60c1954 failed only the `lint, unit, core, system,
compiler` job, before repository tests ran. The Toolchain step printed
`GNU Make 4.3`, then `make: write error: stdout`, and exited 1. The other jobs
completed successfully. The raw failed-job log is retained here.

The step used `set -euo pipefail` with `make --version | head -1`. The reader
can close stdout before the producer finishes, making a version display fail
nondeterministically. Both workflows now use `sed -n '1p'` for version output;
this consumes the entire stream while retaining only the first line for display.
The producer's actual nonzero exit still fails the pipeline. No test, required
gate, toolchain constraint or acceptance threshold is weakened.

Next experiment: the next CI run on the corrected source must complete the
Toolchain step and execute the previously skipped checks. Success requires
those checks' own results; failure retains the first new error for diagnosis.
This correction does not establish any R40 board/performance gate.
