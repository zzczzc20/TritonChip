# R42 board recipe: the 113 original points on the R41 run-3 image

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md) and the shared-card rules of
> [`evidence/r41/SERVERS.md`](../../r41/SERVERS.md). Every number produced by this recipe is a **board** number
> (U280, image `791b8af2…`, measured 100 MHz). Nothing here is typed in; failures are kept.

## 1. How run 3 measured the 113 points (reconstructed from its records)

Sources: `evidence/r41/hbm_bw/g4/run3/board/original113_records.tgz` (`report.json`, `batch.json`, `state.json`),
`PROGRESS.md`, and the registration files that were only on the FPGA host
(`~/r41run3/runtime_prepared/spec/r41/run3/`), now copied to [`run3_registration/`](run3_registration/).
`r3-s2-original-suite.json` there has sha256 `1696ea1b…`, the `manifest_sha256` of the run-3 report.

| Item | Run 3 (2026-10-30) |
|---|---|
| Queue | `tools/r41/run2_board/candidate_queue.py` loaded the image (`program.sh --check`, `--go` after the idle rule); `candidate_resume.py` (`~/r41run3/queue_resume_2`) ran the stages. Stage `original113` = `stage.py` -> `experiment.py` -> step `original` = `board_batch.py --batch original` |
| Manifest | `spec/r41/run3/board/r3-s2-original-suite.json`, derived by `candidate_manifests.py` from `spec/r41/run2/board/r2-s2-original-suite.json` <- `spec/r41/stage2/board/s2-original-suite.json` |
| Points | `spec/r40/suites/existing.json` + `hbm-tier.json` + `large.json`, coverage checked against `hbm.json` (113 unique points), in that order |
| Software tree | `~/r41run3/runtime_prepared`, identity `files:e7b332af…` (R41 branch of 2026-10-30) |
| Interpreter | `~/tc_env/torchenv/bin/python` (venv path kept unresolved; torch 2.14.0+cpu, triton 3.8.0, python 3.10.12) |
| Process environment | XRT/Vitis setup + `TNCA_BOARD=u280` (`auto.board_environment`); manifest env `MAX_JOBS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH={root}:{root}/python:{root}/compiler/python:{root}/tools/asm`, fresh Triton/Inductor/extension caches under the output; ambient `TNCA_*` removed |
| Per-run command | `python -u -m archbench.run --platform tnca --backend fpga-u280-multicore --backend-args {...} --profile core-vrf-sync-v5-x8-g2c4-dp --point P --jobs 1 --seed 0 --warmup 0 --timed N --timeout 28800 --bundle-dir ... --out ... --set TNCA_ACTIVE_CORES=3 --set TNCA_TORCH_DEVICE=native --set TNCA_TRANSFER_POLICY=resident`, one process per run; `board_batch.py` adds `TNCA_OPERATING_CLOCK_EVIDENCE`, `TNCA_DEADLINE=200000000000`, `TNCA_HOST_TIMEOUT=28800` |
| Sampling | `spec/r41/stage1/plan.json` `measurement.long_device_tasks`, mode `device-rule`: a qualification record (cold call untimed, 0 warm-up, 1 timed call); a point at <= 1 s device time gets a formal record of 3 timed calls in a new process, otherwise the qualification record is the formal one (1 sample). Run 3: 113 `-q` runs + 92 `-f3` runs, 6,077 s of process wall time, 1 h 41 min in the queue |
| Boundary | ArchBench protocol 2 (`archbench/PROTOCOL.md` 3, 5.1): `device-cycles-serial-grid`, median of the formal samples, at the qualified 100 MHz |
| Identity first | `r3-s2-identity`: image-source review, `clock_qualify.py verify`, then 3 points (`op.add/tiny/fp16-n1000`, `op.matmul/tiny/fp16-m16-k32-n16`, `model.llama_decode/tiny/fp16-b1-p16`), 1 timed call each |

**Settings the image requires** (all in the manifest `board` block, none is a tuning knob):

- `profile` / `backend_args.image_profile` = `core-vrf-sync-v5-x8-g2c4-dp` (8 barriers, 4 contexts, `m.mma.dp`, snapshot
  fences in the compiler). The hybrid LSU and the line engine are built into the image (`g4/run2/image.tcl`);
  `hybrid_lsu=True` is an argument of the RTL-simulation backend only, the board backend takes no flag for it;
- `backend_args`: `core_count 3`, `memory_ports 3`, `hbm_dma true`, `hbm_bytes 8589934592` (the explicit 8 GiB
  allocation of the `HBM_STRIPED` layout), `device_clock true`;
- memory map: window 0..8 GiB, `local_limit 131072`, `MBG[0:31]` / PC0-PC31, source `xclbin_hbm_map.json` (sha256 `8afe58d9…`);
- qualified clock evidence `~/r41run3/qualified_clock.json` (sha256 `df451e80…`, qualification
  `r41-corrected-candidate-100mhz`, the reviewed entry of `python/tnca/operating_clock.py`), 100 MHz.

**Knobs of every configuration** (the run-3 registration, unchanged): `TNCA_ACTIVE_CORES=3`, `TNCA_TORCH_DEVICE=native`,
`TNCA_TRANSFER_POLICY=resident`. Everything else is the software default.

## 2. The R42 queue

`tools/r42/board_ab.py register` derives two manifests from the run-3 registrations; the changes are listed in the
tool (`CHANGES`) and in each manifest's `registration.changes_against_run3`: new ids and wording, the manifest and
the HBM map are read from outside the measured tree, `estimates` removed, the RTL image-source review step of the
identity manifest removed. Points, order, knobs, sampling, boundary, board block, environment: unchanged.

On the FPGA host `tools/r42/board_ab.py run`:

1. checks the registration hashes, the staged tree identity against the registered one, the image sha256 and the
   author's approval record (`~/r41run3/author_approval.json`);
2. `acceptance_queue.py --check-only` + `preflight_board_manifest.py` (no card access);
3. only when the card holds another image: waits with `tools/r41/run2_board/auto.py` (`IdleTracker`: 15 consecutive
   minutes with no context, no outstanding exec, no client pid, the same loaded UUID and exec count, 5 s poll),
   takes the shared lease `~/r41run3/candidate_queue.lock`, runs `flow/fpga/program.sh --check` and then `--go`
   (`program.log` in the output), verifies UUID and 100 MHz. No retry, no reset;
4. `tools/performance_harness/launches/acceptance_queue.py` (budget 10,800 s) on the identity manifest, then the
   suite: it waits again for 15 idle minutes on the registered image, takes the same lease, and `experiment.py`
   / `board_batch.py` check image, clock and card use before every run and after every step. It never programs.

If the card changes hands the run stops with its evidence; a restart is a new `launch` under a new NAME (or the
same output after inspection, `experiment.py` resumes exactly) and always waits a fresh 15 minutes.

Host layout: `~/r42/<DIR>` software trees (`git archive`, `SOURCE_COMMIT`), `~/r42/reg/NAME` registration + runner,
`~/r42/q/NAME` output (`events.jsonl`, `program.log`, `queue.log`, `queue/<manifest id>/...`).
The guard code always comes from `~/r42/src_base` (unchanged `main` `4b61abd8`); the measured tree supplies
`experiment.py`, `board_batch.py`, ArchBench, compiler and runtime.

## 3. Commands (lead machine)

```sh
T=tools/r42/board_ab.sh
$T stage    4b61abd8 src_base                      # once per software tree; refuses to overwrite
$T register b41-run3-base 4b61abd8 src_base        # writes spec/r42/board/b41-run3-base/
git add spec/r42/board/b41-run3-base && git commit && git push origin R42     # BEFORE any measured run
$T check    b41-run3-base                          # no card access
$T launch   b41-run3-base                          # detached; refuses an unpushed registration
$T status   b41-run3-base                          # poll sparsely (about every 10 minutes)
$T fetch    b41-run3-base                          # evidence/r42/board/b41-run3-base/records.tgz
python3 tools/r42/board_compare.py --new evidence/r42/board/b41-run3-base/records.tgz \
        --out evidence/r42/board/b41-run3-base/comparison.json
```

A candidate (same image, A/B against `b41-run3-base`):

```sh
$T stage    <candidate commit> src_<name>
$T register ab-<name> <candidate commit> src_<name>      # default knobs; add --knob TNCA_X=V only for a diagnostic
git add spec/r42/board/ab-<name> && git commit && git push origin R42
$T check ab-<name> && $T launch ab-<name>
$T fetch ab-<name>
python3 tools/r42/board_compare.py --old evidence/r42/board/b41-run3-base/records.tgz \
        --new evidence/r42/board/ab-<name>/records.tgz --out evidence/r42/board/ab-<name>/comparison.json
```

The candidate commit must be reachable in the lead checkout (fetch the worker branch first). A registration is never
rewritten: a rerun takes a new NAME. A new R42 image needs its own approval, clock qualification and HBM map; then
`IMAGE*`, `APPROVAL`, `HBM_MAP` in `board_ab.py` and the approved list in `acceptance_queue.py` change with it.

## 4. Limits

- One suite pass is about 1 h 45 min of card time plus the idle waits (15 min before a load, 15 min before the queue).
- `acceptance_queue.py` caps a queue at 3 h of execution; a stop on budget keeps its records and resumes.
- Compilation on the FPGA host is single-threaded (`MAX_JOBS=1`); no Vivado, no build, nothing as root.
