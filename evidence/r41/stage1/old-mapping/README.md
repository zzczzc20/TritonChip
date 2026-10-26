# Isolated old-mapping comparison: in progress

This work addresses the remaining Required item
`S1-02.old_mapping_and_conservative_baseline_comparisons`. The frozen product
remains `25094197f6e6`, software
`files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
Stage 1 is **not accepted**. The required index is unchanged pending board data.

## Control construction and numerical validation

`preparation.json` identifies an isolated control with software
`files:b95b380161aebba3f4dee6f4c60ca1c76abb064f5d90863a454d8a3717fa484b`.
It removes only `_OpView.value_users` from the frozen compiler's
`register_blocks.py`, restoring the pre-repair virtual-column SSA lookup.
All other product files match the frozen source. The original directory is
unchanged. This is a diagnostic compiler ablation, not the entire historical
R40 compiler and not a new product default.

`r41s1-old-mapping-control-v1-proof.tgz` retains six independent native-model
processes: old/fixed variants at columns 16 and 32, plus a second K length for
the wide pair. All twelve mm/addmm numerical checks pass on the common standard
Torch/Inductor/Triton/Session path. Inputs, output bindings, binary/configuration
identities and emitted resource plans are retained. `validation-v2.json` checks
the exact cohort and source identities.

| Tile / K | Graph | Old ACC row commands | Fixed ACC row commands |
|---|---|---:|---:|
| 16x16 / 65 | mm | 0 | 0 |
| 16x16 / 65 | addmm | 0 | 0 |
| 16x32 / 65 | mm | 960 | 0 |
| 16x32 / 65 | addmm | 960 | 128 |
| 16x32 / 33 | mm | 576 | 0 |
| 16x32 / 33 | addmm | 576 | 128 |

The narrow executables are byte-identical between variants. Wide fixed code
has zero K-loop ACC reads, writes and zeros; the old control has all three.
Fixed addmm's 128 row operations are outside the K loop and invariant with K,
as the contract permits for its epilogue. Useful dynamic MMA counts agree.
The initial validator's incorrect zero-total-row assumption is preserved in
`validation-initial-failure.json`; it was corrected using the actual loop ledger
and the additional K-scaling check, without changing any numerical acceptance.

Five preparation refusals protect source/report isolation and prevent
overwriting existing evidence. Six validator mutants reject changed source,
input hashes, loop traffic, workload population, per-row numerical status and
dtype. They are recorded in `guard-checks.json` and
`validator-negative-checks.json`. Model cycles remain uncalibrated predictions;
they are not the missing physical comparison.

## Registered board arm

`board-manifest.json` registers all nine original GEMM subset points, their
unchanged hashes, original wide-tile knobs and frozen sampling rule. It uses
the B40 image and 90 MHz qualified clock. The manifest stays outside both
product source trees. The existing completed fixed-wide and conservative
16x16 records will be reused; their severe wide-tile slowdowns remain reported.
The new arm does not promote the wide candidate or overwrite a historical run.

Host directories:

- GB10 source: `~/r41s1-old-mapping-control-v1`.
- GB10 proof: `~/r41s1-old-mapping-control-v1-proof`.
- acf3019 source: `~/r41s1_old_mapping_control_v1`.
- acf3019 queue/output: `~/r41s1_old_mapping_v1`.

The first queue started at 2026-10-26 16:40:52 UTC, PID 2847878. After the
15-minute idle interval it successfully loaded the approved B40 image, but
the experiment runner refused the manifest before workload measurement:
`expected_counter_changes` was missing. That process terminated at 16:56:09.
`board-manifest.v1.json` and `preflight-and-failure-v2.tgz` preserve the original
registration, loader/probe logs and error; no failed record is overwritten.

The corrected `board-manifest.json` (also retained as `board-manifest.v2.json`)
declares the expected counters. The source package now includes the unchanged,
hash-checked board-map reference outside the product hash. The updated
preparation tool reproduces the same control software tree
(`preparation-v2-check.json`). Installed-runner preflight validates real
source/image/clock/map files and reaches the first probe boundary, where a
sentinel deliberately stops execution. It executes no device probe or command.
The original invalid manifest still refuses before output creation.

The corrected waiting queue started at 2026-10-26 21:33:08 UTC, PID 2882868.
`../launches/old_mapping_queue.sh` validates installed source/proof identities
and the successful preflight's manifest hash,
and requires 900 continuous idle seconds, even if B40 is already loaded.
It checks the approved image and qualified clock, then runs the registered
manifest through `experiment.py`. Each board configuration retains the existing
UUID/clock/usage checks. Waiting is bounded to 24 hours; the original manifest's
12-hour execution budget is retained. No full image build is requested.

Inspect the live PID, `queue.log`, `experiment.log` and
`out/old-mapping/state.json` before taking the next action. A timeout in
observation is not permission to restart. Numerical and source/timing checks,
old/fixed and fixed/narrow comparisons, report updates and frozen acceptance
reproduction remain necessary after the physical run.
