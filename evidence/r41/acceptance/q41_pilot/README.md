# Q41 full-model pilot preparation

The existing run-3 queue does not cover the required BF16 Qwen3.5-2B Q41
workloads. A bounded prefill-plus-one-decode pilot is being prepared; it is
not a full Q41-core result or Stage 2 acceptance. The frozen 24-layer model,
input IDs and independent FP32 state references are used unchanged.

On GB10 CPU, the final pilot script passed Q41-core steps 0 and 1, comparing
49 tensors at each step against the frozen FP32 reference. Logits relative L2
were 0.01259 and 0.01197; maximum state relative L2 was 0.01328. Six extra
StaticCache length tensors per step were checked explicitly. This is CPU
evidence only; no TNCA pilot has yet run. The final script SHA-256 is
`fb6c519b678f0b6f7bab0cfc005676e2cc0b60c15c1902ef01f717c906d098cb`.

The first CPU attempt correctly refused the installed transformers 5.20.2
because its source hash differed from the independent reference. The second,
using the pinned official source, reported StaticCache length metadata not
present in the DynamicCache reference. The third and later attempts checked
those lengths separately and passed the original numerical thresholds. The
raw failure and passing records are in [`cpu_pilot_attempts.tgz`](cpu_pilot_attempts.tgz)
and [`cpu_pilot_final.tgz`](cpu_pilot_final.tgz).

The exact checkpoint, all 22 FP32 reference files, and the three pinned
transformers source files are staged on the FPGA host and hash-verified:
[`staged_inputs.tgz`](staged_inputs.tgz). The pinned transformers sources also
imported successfully in the FPGA Python environment without accessing the
card. The pilot refuses a busy card, wrong image or clock, mismatched backend,
or an occupied candidate queue lock; five simulated guard tests passed
([`board_guard_tests.log`](board_guard_tests.log)). It rechecks card idleness
immediately before opening the device and does not program an image.

The preregistered pilot is two steps with a one-hour limit:
[`r3-q41-bf16-pilot.json`](../../../../spec/r41/stage2/board/r3-q41-bf16-pilot.json),
SHA-256 `84288a95be50f4f9b725062281cdde248dda7c850baa5412cbbc7418b9199d02`.
Its waiter requires all twelve preceding queue stages to finish and verifies
their registered result hashes before waiting for the matching idle card. It
stops on an upstream or pilot failure and retries only a card/lock contention
detected before device execution. Five waiter evidence-validation tests pass:
[`waiter_tests.log`](waiter_tests.log). No source or registration is changed
in the already-running board campaign.
