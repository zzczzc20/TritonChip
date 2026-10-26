# R41 acceptance status

2026-10-26 UTC — milestone 1: acceptance ownership established at origin/r41
691d0dd in the dedicated r41-accept worktree. Read HANDOFF.md and SERVERS.md.
No card access, board job, or full image build has been started.

Stage 2 and R41 are not accepted. Evidence inventory confirms the 100 MHz
run-1 image, original-suite results, 8 GiB cases and corrected endurance result.
Full Q41-core/tail, XL-core, locality comparisons and independent reproduction
still need explicit judgments. Raw board evidence is being collected read-only.
Stage 1 closeout and the r41-hbm-bw workstream remain with their existing owners.

Milestone 2: raw board JSON/JSONL and physical audit collected without device
access. `python3 evidence/r41/acceptance/closeout.py` validates the 113 frozen
original point hashes (including all 24 large points), qualified image/clock,
source check, 100 FP8 and 16 blockwise cases, nine high-address cases and the
corrected endurance record. Archive/member hashes are in board_index.json.
The complete 113-point Stage 1 comparison is now available: ratio of summed
median device times 1.09673x; pointwise geometric mean 1.09732x. This is a joint
hardware/frequency/software migration comparison, not a compiler-only gain.
The old 96-point comparison remains historical evidence.

Milestone 3: all 94 Stage 2/DoD and supporting study/common obligations judged:
24 passed, 2 failed, 60 not measured, 8 blocked. Counts include aggregate and
study rows; they are not 94 independent functional tests. Eight numbered
S2-01 candidate checks pass. The >65,535-program requirement is closed from
existing standard Torch launch maps (maximum 4,194,304 programs).

Both requested reports are written in professional English. Stage 2 and R41
remain not accepted. Frozen Stage 1 snapshot remains 59/61 passes, one not
measured and one blocked; its owner retains the remaining comparison.
P0–P8 in closure_plan.md give bounded work, dependencies and resource decisions.
No board run or full image run was started. No new remote branch was created.

Validation: closeout.py passed raw board/physical/endurance checks and the
complete 113-point comparison; six Stage 2 checker refusal tests passed;
`python3 -m tools.docmaint check` passed (85 documents, acyclic).
`make r41-stage2-acceptance PYTHON=python3` and
`make r41-acceptance PYTHON=python3` both reject acceptance as expected, with
zero evidence-integrity errors. Overall target uses the archived checker
judgment of the frozen Stage 1 tree; it does not rerun that product in this
merged acceptance worktree. judgment.json preserves the aggregate result.

Final review retained the illegal-alias negative-test gate as not measured:
high-bit anti-alias tests pass but do not establish overlapping-storage
ownership refusal. The compound requirement is not upgraded from partial evidence.
