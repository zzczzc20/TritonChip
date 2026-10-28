# Non-regression gate after the r41 rectification

The audit (`evidence/r41/hbm_bw/AUDIT.md`, §"For the lead", F26, R16, R17) found that the merge e4770b6 brought default-path RTL changes into `r41` (core, LSU, NIs, NUCA slice, outlets; every new feature default-off) without a `make compiler`, `default_change_audit.py` or 113-point record, and the stream removal (60b7ab7, f2f3dc6) then touched the runtime. This directory records that gate (discipline §2.4.1).

## Identity and method
- Host: GB10 (`gx10-6646`), `source ~/tc_env/env.sh`, at most six cores (two RTL streams plus a four-job audit), 2026-10-28T08:28Z–12:10Z.
- **Head:** `r41-fix` 116f747 (rectification code changes on top of origin/r41 f2f3dc6), software tree `files:c98832e9…45e8`.
- **Base:** e6b613d, the first parent of e4770b6 (r41 just before the merge), software tree `files:ccb76134…196b`.
- **Final:** 2abfd07. Its product code (`python`, `compiler`, `rtl`, `archbench`, `spec`, `tb`, `flow`, `fpga`, `xls`, `Makefile`) is identical to head; only tests, documents and evidence differ. Tests changed after head were rerun on it.
- Trees were staged with `git archive`. The torch environment carries an editable `tnca` install from another tree (`~/tritonchip`), so every lane sets `PYTHONPATH` to its own tree and records the resolved import paths (`logs/{head,base}/run.txt`). A first attempt without this (`logs/attempt1/`) imported the other tree and is kept, not counted.
- Scripts: `gate.sh` (RTL lanes per tree), `post.sh` (strict `make compiler` after the RTL lanes), `aux.sh` / `aux2.sh` / `aux3.sh` (audit, unit tests, FMA miter), all inside `gate-logs.tgz` under `logs/`. `R1_STRICT=1` throughout, so a skip is a failure.
- `compare_lanes.py` compares the head and base result lines (PASS/FAIL/cycles/mismatch, paths and timestamps normalised) and wrote `lane_comparison.json`.

| Archive | sha256 |
|---|---|
| `gate-logs.tgz` (every lane log, exit codes, both audit reports) | `0ccf2840e23f29d9a9fa162dcb13461e56401e3fc925137cd338071beba1574a` |
| `rerun-multicore-dma.tgz` | `d350f40eaff766dd2c74bb95e4cdb4e8f6527276b467db7f9b0eef13cb8f26b6` |

## Results

**Default-change audit (113 points).** `default_change_audit.py run` on head and on base, then `compare`: **113 unchanged, 0 changed, 0 not compared** (`logs/aux/default_change_audit.json`). Against the r41s1 control (`~/r41-fp8-audit/run-ctrl`) the result is the same (`default_change_audit_vs_r41s1.json`). This covers the merge, the stream removal and the default-off compiler knobs of R16 and R17: the compiled code text and launch structure of all 113 original points are unchanged. It does not measure numerics or performance.

**`make compiler`** (head, strict, after `make tnca-opt` with the pinned LLVM 20): ISA checks pass; compiler-unit **157 tests, 0 skipped**; compiler-ir **6 files, 6 RUN cases** pass (`logs/head/compiler-after-core.log`). The first strict run (`compiler.log`) failed only because `sim/build/tb_core.vvp` had not been built yet, so 53 RTL-backed tests counted as skips; the standalone `compiler-ir.log` failed only because `LLVM_PREFIX` was not passed (FileCheck not found). Both are kept.

**RTL regressions,** head and base, all exit 0 on both trees (`lane_comparison.json`):
- `core`, `core-slowmem`, `core-ooo`, `core-vrf` (40 PASS each), `core-r39hw`, `core-r39sw`, `core-r39sw4` (82 each), `core-r39sw5` (88), `r41-core-boundary` (45), `top` (4), `r38-xpose-rtl` (3), `r41-xpose8-rtl` (7), `lsu-reqreg` (5), `fma-deep`, `lint`, `unit`, `r40-multicore-shell` (20), `r40-grid-dispatch` (359), `r40-cache-maintenance` (26), `r40-coherent-host` (5), `r40-barrier-epoch`, `r40-core-parameters` (7), `compiler-rtl`, `triton-e2e` (8), `r40-pipeline-checks` (31), `multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top` (455 programs).
- Result lines, **including every cycle count, are identical** between head and base for all of these except three, explained below.
- `r40-line-axi`: all 35 base configurations reappear unchanged in head; head adds 43 configurations (new data widths) and 8 outlet streaming cases from the merged default-off outlet work.
- `unit`: the assembler round-trip covers 58 programs in head against 56 in base (two new test programs), 0 mismatches in both.
- `r40-multicore-dma`: PASS on both, but the per-port line counts and the program-to-core split differ between runs. A rerun of each tree (`rerun-multicore-dma.tgz`) shows the base reproducing head's first counts (347/283/256 read lines) and head producing 353/286/256: the counts vary from run to run in both trees, so the difference is not attributable to the merge.

**FMA miter** (`evidence/r41/stage2/candidates/F03/miter`), head FMA against the generated FMA before F03 (`f3db5c5^`, the integration's reference), Verilator 5.020: **210,000,000 vectors × 2 latencies, 0 mismatches** (`logs/aux3-final/fma-miter-vlroot-30m.log`; also 7,000,000 × 2 with seed 1). The two earlier miter attempts failed on gate staging (a reference directory without the pre-F03 module; `VERILATOR_ROOT` from `env.sh` has no `bin/verilator_bin`) and are kept.

**Unit tests touched by the stream removal** (runtime, multicore MMIO, U50 link, torch copy, flow): 32 test files on head, 31 on base (`tools/r41/hbm_layout_tests.py` does not exist there), `logs/aux2-{head,base}/`. Head: 27 pass, 5 fail; base: 28 pass, 3 fail.
- Fail on both trees, so not caused by the merge or the removal: `tools/r36/fpga_tests.py` (R36-FPGA-02/03: this host's yosys rejects the constant function call at `rtl/core/tnca_tensor.v:564`, a file the merge did not touch); `tools/r2/memory_tests.py` (MEM-01/02/04 are permanently NOT RUN: they need a cached memory model and an injection hook); `tools/r40/large_grid_tests.py` (stale fixture, see below).
- Fail on head only: `tools/performance_harness/test_hbm8.py` (three tests) — the merged striped-layout code (6e4bebf) reads the capability register in `hbm_open`, and the test's accessor stand-in could not answer register reads; `tools/performance_harness/test_stage2_acceptance.py` — the checker's hash of `evidence/r41/SERVERS.md` was stale on origin/r41 since 208fbb1.
- Fixes (2abfd07 and 7775a9c), generic and with unchanged assertions: the hbm8 stand-in answers register reads with 0 (the B40 image it models has no striped layout); the large-grid fixture sets `_measurement_owner`, which `Session.__init__` sets since f5ff452; the index re-hashes `SERVERS.md`. All three pass on the final tree (`logs/aux3-final/`).
- Nothing removed with the stream engine was still referenced (`bash -n flow/fpga/link.sh` passes; no remaining reference to the removed modules outside archived evidence).

## Verdict
The gate passes for the merge and the stream removal: 113 of 113 original points compile to unchanged code, `make compiler` passes strictly, every RTL regression passes with cycle counts identical to the pre-merge base (one lane with run-to-run variable counts on both trees), and the FMA miter shows no mismatch. Two stale test fixtures and one stale evidence hash were fixed. Pre-existing failures (`r36` yosys elaboration, `r2` NOT RUN cases) remain and are not caused by this work.
