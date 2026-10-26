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
