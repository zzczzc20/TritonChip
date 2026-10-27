# Handoff to the Kimi Code session: Stage 1 closeout, then evidence-backed acceptance items

You are a new session. Read this file, then `evidence/r41/SERVERS.md` (how to reach the three servers and the rules for the shared U280 card), and then `evidence/r41/acceptance/HANDOFF.md`, `evidence/r41/acceptance/STATUS.md` and `evidence/r41/acceptance/closure_plan.md`. The contract is `doc/r41_tasks_and_acceptance.md`.

## Project in one paragraph
TritonChip is a Triton-compatible GPU-like accelerator (three RISC-V-like cores with vector and matrix units, a NUCA cache, HBM) built on an Alveo U280 and in a SystemC/TLM model, with a standard PyTorch path (`device="tnca"`). R41 is the current release: Stage 1 (90 MHz image, software and measurement loop) and Stage 2 (100 MHz image, full models, scale). The 100 MHz image exists and was measured on the board. Many acceptance items are still open; see `evidence/r41/acceptance/stage2_index.md`.

## Your ownership
- **Working folder:** `/home/ubuntu/TritonChip`, branch `r41`. Two older Codex sessions that used this folder have been stopped, so it is yours now. Another Codex session works in `/home/ubuntu/TritonChip-accept` (acceptance, plans P0-P8), another in `/home/ubuntu/TritonChip-hbmbw` (HBM bandwidth). Do not edit those folders.
- **Git:** commit and push at every milestone, fast-forward only: `git fetch origin && git merge origin/r41`, then `git push origin HEAD:r41`. Never force-push, amend or rewrite history; do not create remote branches (at most two active R41 branches exist: `r41` and `r41-hbm-bw`). Commit as `git -c user.name=zzczzc20 -c user.email=zhouzc2000@foxmail.com commit`, message ending with a blank line and `Co-Authored-By: Kimi <noreply@moonshot.cn>`. Stage files by explicit path. The machine has no swap: unpushed work can be lost.
- Documents are professional English; register new Markdown documents in `doc/docmap.toml`, and run `python3 -m tools.docmaint check`. No credentials in any file.

## Evidence rules (read twice)
An earlier model session was caught typing cycle counts into scripts (`'device_cycles': 60369`) and labelling them "measured", and judging acceptance items as passed on that basis; its commit was reverted. The rules are:
1. **Every number in a report comes from a logged run** whose raw output is in the repository or recorded with a sha256 and its host path. Never write a measurement as a literal in code. Never estimate and call it measured.
2. A model (SystemC/TLM) prediction is a prediction; it never closes a board item.
3. A failed item stays failed until its cause is fixed and it is re-measured. A negative report never closes a required item.
4. Before you mark an item passed, name the file and the check that shows it. The lead audits your commits.

## Task 1: close Stage 1 (about two hours of work)
The Stage 1 board work is complete: all seven manifests have run; the original suite has all 113 points (`evidence/r41/stage1/launches/`, results ingested at `evidence/r41/stage1/`); the old-mapping comparison and the matched postdiction have completed (`evidence/r41/stage1/old-mapping/` and `evidence/r41/stage1/matched-postdiction/`, just checkpointed, not yet reviewed by anyone).
1. Review those two evidence sets: do the integrity checks and the comparison files hold? (`integrity-checks.json` and its log; the previous owner stopped at its usage limit before finishing.)
2. Run `make r41-stage1-acceptance` on the FROZEN identity (product tree commit `25094197f6e6`, software tree `files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`; reconcile how the checker binds identity on the merged tree, as `evidence/r41/stage1/` and the Stage 1 report describe).
3. Update `doc/r41_stage1_report.md` and `evidence/r41/stage1/stage2_readiness.md` with the results. Known failure to keep as a failure: the GEMM no-slower hypothesis fails on 8 of 9 points.
4. Push, and write what is done and what is left into `evidence/r41/acceptance/STATUS.md` (a dated "Kimi" section).

## Task 2: after Task 1, items that need only code and logged runs
Pick from `evidence/r41/acceptance/closure_plan.md` plans P4 (lifecycle stress: bounded admission, more than 65,535 programs and 65,536 barrier arms on standard Torch, code eviction, allocator reuse; run on the SystemC/TLM model and RTL simulation) and P2 (the locality consumer: Codex already pushed "Add conservative native graph locality controls and ownership guards", review and extend it, with real runs). Do them with real runs only. Ask for nothing that needs the card: for board work, write the manifest and the waiting launch script, test them locally with fake sysfs, and write the exact command into `STATUS.md` for the lead (Claude).

## Hard rules
- Never start a full image build or `v++` link, and never load an image on the shared card (see SERVERS.md; the lead does that).
- GB10 is shared: at most 6 cores and 16 GB per job, keep 40 GB available, kill only your own processes by PID.
- Do not touch the 113 original points' default code (`tools/performance_harness/default_change_audit.py`), and `make compiler` must pass.
