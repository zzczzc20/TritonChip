# Stage 1 board evidence

Read-only snapshot from `acf3019:/home/zzhouch/r41s1_board_25094197f6e6`, retrieved 2026-10-26 UTC.
Product commit: `25094197f6e6`; software tree: `files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
Image: B40 `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, 90 MHz, three cores, 4 GiB.

`finished.tgz` contains the six completed manifests' reports, state, batches, formal records and comparison outputs under `out/board-*`.
`index.json` records every archive member's size and SHA256 plus the archive digest.
`host-raw.sha256` preserves hashes of the host queue log and raw logs/JSONL; paths there are relative to the host root above. The line immediately after the queue digest records its byte size. Large raw logs, caches and compiler dumps remain on the host and are excluded from git. The archive is a selected evidence bundle, not a complete execution backup.

Extract locally with `tar xzf evidence/r41/stage1/board/finished.tgz -C /path/to/evidence`.
A completed experiment records command completion; its hypothesis and contract gates require separate judgement.

`board-original-suite` is **not-measured**. The interrupted first attempt is excluded from this bundle; the lead schedules its rerun. No queue, script or device state was changed during collection.

## Local judgement and frozen acceptance

`executables.tgz` retains the actual executable artifacts from the GEMM and
prospective formal bundles. Its SHA256 is in `derived-index.json`.
`closeout.py` validates formal records, verifies executable identities against the
sealed predictions and recomputes residuals and ratios into `metrics.json` and
`*-residuals.json`. It never reads a device or runs a model. Use:

```sh
python3 evidence/r41/stage1/board/closeout.py --frozen-root /home/ubuntu/r41s1-frozen-25094197f6e6
bash evidence/r41/stage1/board/run_acceptance.sh /home/ubuntu/r41s1-frozen-25094197f6e6
```

Create the detached worktree first if absent:
`git worktree add --detach /home/ubuntu/r41s1-frozen-25094197f6e6 25094197f6e6`.
The acceptance script copies Stage 1 evidence and the report, extracts the archive
under `board/out/`, verifies the frozen tree and contract digest, and runs the
unchanged Make target. Its nonzero exit is retained; do not replace it with success.
The source branch's merged contract remains untouched. The [report](../../../../doc/r41_stage1_report.md)
explains the evidence-only overlay and indexed-document mismatch.

## Original-suite ingest placeholder

Status: **not-measured**. Wait for the lead to confirm the rerun completed.
Then fetch only its evidence using this file-read-only command (no experiment
execution, queue mutation or card access):

```sh
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
"$S/fp" 'cd ~/r41s1_board_25094197f6e6 && tar czf - out/board-original-suite/report.json out/board-original-suite/original/state.json out/board-original-suite/original/batch.json out/board-original-suite/original/formal.jsonl' > evidence/r41/stage1/board/original-suite-rerun.tgz
```

Keep the interrupted attempt and failure history on the host. Hash/index the new
snapshot before extracting it. Require `batch.json` status `complete`, no rejected
runs, 113 unique frozen points (77 native, 12 HBM-tier, 24 large), and the frozen
software/image/clock identities. Apply `stage1_acceptance.py` with
`coverage: original_suites` to these actual formal records. Compare section 6.5's
matched performance set and report all >5% regressions and confirmed >10%
new-default regressions. Archive the comparison and update the report and index
before rerunning frozen acceptance. A complete numerical suite alone does not
approve the nonregression row. Never substitute the 100 MHz Stage 2 suite.
