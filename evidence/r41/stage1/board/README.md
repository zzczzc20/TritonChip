# Stage 1 board evidence

Read-only snapshot from `acf3019:/home/zzhouch/r41s1_board_25094197f6e6`, retrieved 2026-10-26 UTC.
Product commit: `25094197f6e6`; software tree: `files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
Image: B40 `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, 90 MHz, three cores, 4 GiB.

`finished.tgz` contains the six completed manifests' reports, state, batches, formal records and comparison outputs under `out/board-*`.
`index.json` records every archive member's size and SHA256 plus the archive digest.
`host-raw.sha256` preserves hashes of the host queue log and raw logs/JSONL; paths there are relative to the host root above. The line immediately after the queue digest records its byte size. Large raw logs, caches and compiler dumps remain on the host and are excluded from git. The archive is a selected evidence bundle, not a complete execution backup.

Extract locally with `tar xzf evidence/r41/stage1/board/finished.tgz -C /path/to/evidence`.
A completed experiment records command completion; its hypothesis and contract gates require separate judgement.

`board-original-suite` is complete: 113/113 frozen points, 202 accepted runs, no failed final runs. Its separate `original-suite-rerun.tgz` includes the interrupted record and pre-retry state. No queue, script or device state was changed during collection.

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

## Completed original-suite ingest

`original-suite-rerun.tgz` is the read-only snapshot of `out/board-original-suite`.
`original-suite-index.json` hashes the archive and every file member before extraction.
It contains `report.json`, `original/{batch.json,state.json,formal.jsonl,runs/}`,
`original/runs.interrupted/` and `original/state.json.before-retry-of-interrupted-point`.
Extract it alongside `finished.tgz`; extracted `out/` is excluded from git.

The first attempt completed 112 points. The qualification call for
`op.attention_decode/large/fp16-b1-h32-l2048-d64` was interrupted by another
user reloading the card (2026-10-25 19:33:02–19:33:13 UTC). The pre-retry
state preserves its rejected `q` entry. That raw record was retained under
`runs.interrupted/`. On the same frozen software and 90 MHz B40 image, only
that point was requalified with seed 0, warmup 0, timed 1 on October 26
09:04:21–09:04:40 UTC, followed by a separate formal timed-3 process at
09:04:40–09:04:59 UTC. The other 112 points were reused, not rerun.
The accepted state has 202 successful runs; the historical rejected attempt
is preserved and excluded from final numerical/performance statistics.

Run `python3 evidence/r41/stage1/board/original_closeout.py FROZEN_ROOT`
after extraction. It validates every formal record with the frozen checker,
checks 113 unique original hashes, complete batch status and no rejected final
runs, and writes `original-suite-comparison.json`. Independent numerical
references pass; maximum error is 5.8543% of tolerance.

The B40-OBS reference is `evidence/r40/v3/R40-08.attention/board_native_fe1b/`:
77 native points, 12 HBM points from `hbm14` (exclude its two duplicated large
attention points), and 24 from `large24`. All software/image/configuration and
point hashes are checked. Median serial-grid cycles exclude cold calls; both
products run at 90 MHz. All 113 points match: geometric mean R41/R40 0.998400,
sum ratio 1.000322; no >5% or >10% regressions. Section 6.5 device-time
nonregression passes the 5% budget. Baseline points have one sample, so these
are descriptive comparisons, not confidence intervals. Host latency and graph
makespan nonregression are not established by serial-grid measurements.
