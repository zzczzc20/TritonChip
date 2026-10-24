# Prospective model predictions v2 (sealed before any board run)

The v2 predictions replace `../prospective-prediction.json` (v1) for the Stage 1 board batch. v1 stays as
history. It is bound to tree `files:b3f03a89…` and artifact `b7cf71e0…`, so it cannot describe the code the
board will run.

**Identity.**
- Product tree: r41s1 `c2c96d3`, `files:8c71390…`. The model was rebuilt on GB10 from a `git archive` of that
  commit (`tools/build.json`).
- Model: `tnca-chip` sha256 `72b6c58c…`. Ramulator2 is pinned at `c6c7a5de…`.
- DRAM paths: fast is `r40-board-m2.cfg` (dram_path 0); slow is `r40-board-m3.cfg` (dram_path 1, Ramulator2
  with the per-command fixed point).
- Address map: the B40 BO/PC map, `TNCA_DRAM_MAP_V1` sha256 `a1019970…`. It is `./map/map.txt` in
  `../physical-dram-map-v2-evidence.tgz`: window base 0 at logical 131072, PC0–PC15 at 256 MiB each
  (`../../address_map.md`). The slow path also uses the controller's real address mapping (`2cf7b05`).

**Configurations.** The knobs are exactly those of `spec/r41/stage1/experiments/board-prospective.json` and
`board-gemm-tile.json`; every results row's knobs were verified against its record (`run_graphs.py`).
- prospective: `op.matmul/medium/fp16-m128-k192-n128`, three resident cores, one port and one outstanding line,
  pipeline 2, at TNCA_GEMM_TILE 16,32,16 and 16,16,16;
- GEMM subset: the 9 points of `spec/r40/suites/R40-03.gemm-reuse.json` × {16,16,16; 16,32,16} with the fe1b
  B40-S knobs (one core, per-kernel transfers, pipeline 1).

**Protocol.** `run_graphs.py` with no warm-up and one timed call. The prediction is the model graph makespan of
the steady call, as the board's formal records measure it. The cold call is timed, except on the three HBM-tier
points, where it runs untimed (recorded per entry as `untimed_cold_call`). Their ~100 MB streams make the 48 KiB
of cache state immaterial, and this halves their simulation time. All runs were confined to 4 cores. At most two
HBM-scale model processes ran at once. A watchdog enforced total RSS ≤ 35 GB and available memory ≥ 40 GB,
killing by PID only.

**Status.**
- Each entry is `sealed`, `sealed-inconclusive-fixed-point`, `not-run` (with its reason) or `failed`. Nothing
  was dropped.
- An inconclusive slow fixed point keeps its cycles but is not a qualified prediction.
- A slow HBM run estimated above 3 h (its fast-path model seconds × the largest measured slow/fast ratio) was
  not run. Started slow runs had a 3 h limit.

**Comparability.** A residual needs the board to execute the same programs. The test uses each measured launch's
executable identity: the artifact segments plus the manifest's `execution` and `resources` sections. The
artifact file hash also covers the run directory's source path; the fast and slow runs of one configuration
differ in it while their code is identical. The measured-call artifacts are kept in `artifacts/`.
`tools/performance_harness/prediction_residuals.py` computes fast and slow residuals per configuration.

`*.record.json` are the model records the entries bind by sha256. `tools/` holds the driver, scheduler, watchdog
and seal scripts, together with the build, group and scheduler state.

## Outcome

**Seal times (both before any board run of this tree).**
- `../prospective-prediction-v2.json`: 2026-10-21T10:55:14Z. A first seal at 10:51:50Z lacked the per-launch
  program identity. It was replaced before any board run and was never committed.
- `../gemm-subset-predictions-v2.json`: 2026-10-21T13:07:45Z. A seal at 13:06:12Z named its four matmul/medium
  records like the prospective point's and overwrote them. It was replaced 90 s later, with `gemm-`-prefixed
  record names, before any board run and before any commit. The prospective records were restored from the run
  outputs, byte-identical to their sealed hashes.

**Statuses.**
- Prospective: 4 entries; both slow ones are inconclusive fixed points.
- GEMM subset: 36 entries.
  - 18 fast predictions are sealed.
  - 16 slow predictions are sealed with `inconclusive-fixed-point`: some commands' request sets or latencies did
    not converge within `dram_max_passes`. Their cycles are recorded but are not qualified predictions.
  - 2 slow predictions are not run: candidate linear/hbm and matmul/hbm at 16,32,16.
    - The pre-stated rule estimated them at 4.1 h and 6.5 h: fast model time × the largest measured ratio,
      6.86, from gemv/hbm.
    - Using the ratio measured later on each point's own 16,16,16 run, they would be about 2.9 h and 3.9 h.
      Both estimates are in `tools/hbm_sched.json`.

**Machine limits.**
- The watchdog saw a peak total RSS of 11.55 GB, at least 94.5 GB available, and killed nothing
  (`tools/watchdog-summary.txt`).
- One HBM fast run was preempted after 80 s so the longest slow run could start first. It was restarted from
  scratch. The preemption is recorded in `tools/hbm_sched.json`.

**Software tree.** Pointing `spec/r41/stage1/experiments/board-{prospective,gemm-tile}.json` at these files
changes archbench's software tree, because `spec/` is part of it. The board will therefore record a tree other
than `files:8c71390…`. Residuals are judged on program identity, not on tree equality.
