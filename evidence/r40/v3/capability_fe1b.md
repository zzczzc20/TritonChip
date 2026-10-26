# R40 v3 board-dependent capability gates on the fe1b delivery identity

Delivery identity (`contract_v3.delivery(plan)` through `doc/r40_v3/identity_amendment.json`):
product tree `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`, image
`a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, measured 90 MHz (clock evidence
`qualified_clock.json`, sha256 `8ad4b3bd…`). Every gate below is judged on evidence bound to exactly this
identity. 80dd/bc5a evidence stays in its `*_80dd` / `history_*` directories and is not an input.

## Sources

| Source | Where | Identity | State |
|---|---|---|---|
| Functional re-collection (`launches/x2_launch.json`, `launches/recollect_x2_controller.py`, sha256 `309b1e9d…`) | FPGA host `~/tnca_v3_recollect_x2`, outputs `~/tnca_runs_r40_v3x2`, source `~/tnca_src_r40_v3x2` | fe1b (live source tree re-hashed on the host 2026-10-20: fe1b) | `complete-review-required`, all 30 stages exit 0 |
| CUDA-host conformance (`launches/cuda_fe1b_launch.json`, controller sha256 `35c32b3b…`) | GB10 `~/r40v3_cuda_fe1b`, source `~/r40v3_capgate_fe1b` | fe1b (live source tree re-hashed on GB10: fe1b) | `failed-preserved`: storage-owner-unit, signed-copy-range, native-recovery exit 0; cuda-device-conformance exit 1 (15/16). Not rerun |
| CI run 39281384427 (`ci_39281384427/`) | GitHub Actions, head `3b73efb`; the job checked out merge commit `108b707` of `3b73efb` into `main` `b5c606d`, an ancestor of `3b73efb`, so the tested tree is `3b73efb`'s | fe1b | success, all 12 R40 lanes the memory-system auditor reads |
| GB10 delivery-tree steps (`memsys_gb10/`), semantics matrix and journal replay (`R40-06.semantics/`), RTL simulation chains (`R40-02.stress/sim/fe1b_sim1`) | GB10 | fe1b | unchanged, already committed |

The raw files were streamed from the hosts with `tar czf -` and their sha256 re-checked against the host copies.
Each `*_fe1b` directory has a `SHA256SUMS` (`sha256sum ./file` format). Gzipped copies carry a
`manifest.json` / `model_report_manifest.json` with the host path, raw sha256 and raw size. Compiler bundles
(141 MB native77/HBM14, 43 MB large24) stay on the FPGA host. Their per-file sha256 listing is in
`R40-05.multicore/board_audit_fe1b/bundles_sha256.txt.gz` (11 478 files).

## Evidence directories

| Directory | Contents | Read by |
|---|---|---|
| `R40-02.zero-transfer/board_queue_fe1b/` | queue `controller.py`, `state.json`, `commands.json`, every stage log (`graphs24-point00..23.log`, …), `snapshot_harness/` (the queue's `frozen_residency_tests.py`, `native_model.py`, `bucket_decode.py`), and `full-original-functional-delivery/` (the nested `tools/r40/conditional_delivery.py` run: `delivery.json` and its stage logs: checkpoint, native77, HBM14, `frozen-original-model.log`) | v3_r40_02_gates, v3_r40_08_gates, v3_conformance_docs (no-CUDA) |
| `R40-08.attention/board_native_fe1b/` | `u280_{tiny,small,medium}.jsonl.gz`, `hbm14.jsonl.gz`, `large24.jsonl.gz`, `delivery.json.gz`, `manifest.json` | v3_r40_08_gates |
| `R40-08.decode/board_model_fe1b/` | `model_report.json.gz` (16.2 MB raw) + `model_report_manifest.json`, on-board `bucket_model_capability.json`, `frozen-original-model.log`, `full-bucket-model-capability-audit.log`, `qualified_clock.json` | v3_r40_02_gates, v3_r40_08_gates, v3_memsys_gates, native_inventory |
| `R40-04.hbm/board_run_fe1b/` | the memory-system auditor's `--board-run` layout, uncompressed because it reads them that way: `checkpoint/report.json`, `native/u280_*.jsonl`, `hbm14.jsonl`, plus `window_samples.json` and `large24.jsonl` | v3_memsys_gates, native_inventory |
| `R40-04.hbm/memory_peaks.json`, `capacity_plan.json` | derived by the R40-04.hbm audit (117 rows, peak `model.llama_prefill/large/fp16-b1-s512` 3 098 945 024 B, headroom 1.3859 in the 4 GiB window) | written by v3_memsys_gates |
| `R40-05.multicore/board_audit_fe1b/` | on-host audit report `multicore_capability.json`, stage log, `command.json`, the gate record the host wrote (`host_gate_evidence/R40-05.json`), `bundles_sha256.txt.gz`, `crosscheck_host_audit.py` and its output | see R40-05.multicore below |
| `R40-01.device/native_inventory_fe1b/` | on-host stage `final-native-inventory-evidence` (log, command, host gate record) and the delivery run's `native77-audit.log` | reference only; the gate is re-judged locally |
| `R40-01.mixed-device/no_cuda_host_fe1b.json` | no-CUDA-host conformance document (`tools/r40/v3_conformance_docs.py`) from the recollect_x2 stage `complete-no-cuda-host-conformance` | device_conformance_acceptance |
| `R40-01.mixed-device/cuda_host_fe1b_failed/` | raw copy of the failed GB10 CUDA-host run (`controller.py`, `state.json`, `commands.json`, `controller.out`, every stage log) | reference; preserved failure |
| `ci_39281384427/` | `run.json` (`gh run view --json conclusion,createdAt,databaseId,event,headBranch,headSha,jobs,status,updatedAt,url`) and the 12 lane logs (`gh api repos/zzczzc20/TritonChip/actions/jobs/<id>/logs`) | v3_memsys_gates |

## Commands and verdicts

Every auditor ran from the repository root on the delivery checkout (local product tree = fe1b), first in
its dry-run form (`--dry-run`, or without `--evidence` for the v2-era tools), then for real.
`D=files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`,
`I=a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`, `E=evidence/r40/v3`.

| Gate | Verdict | Checks | Auditor |
|---|---|---|---|
| R40-01.device | capability-pass | 79/79 | (4) |
| R40-01.mixed-device | capability-pass | 32/32 (no-CUDA half 16/16; CUDA half 16/16 on attempt 4) | (6) |
| R40-02.zero-transfer | capability-pass | 4/4 | (1) |
| R40-04.hbm | capability-pass | 25/25 | (3) |
| R40-04.robustness | capability-pass | 30/30 | (3) |
| R40-05.multicore | capability-pass | 94/94 | (5) |
| R40-05.dispatch | capability-pass | 31/31 | (3) |
| R40-06.nuca | capability-pass | 27/27 | (3) |
| R40-08.attention | capability-pass | 7/7 | (2) |
| R40-08.decode | capability-pass | 5/5 | (2) |
| R40-09.suites | capability-pass | 9/9 | (3) |

(1) R40-02.zero-transfer. The auditor also re-judges R40-02.stress from the unchanged `R40-02.stress/sim/fe1b_sim1`
(still capability-pass, 8/8).

```
R40_CONTRACT=v3 python3 tools/r40/v3_r40_02_gates.py [--dry-run] \
  --board $E/R40-02.zero-transfer/board_queue_fe1b \
  --model-report $E/R40-08.decode/board_model_fe1b/model_report.json.gz
```

All 24 frozen graphs: 48 resident calls with 0 tensor bytes, 48 per-kernel staging controls moving 142 349 696 B,
3 to 84 kernels per graph. Full model: 65 steps, 5 016 steady launches, weights uploaded once (221 267 968 B).
RTL controls `residency-controls` and `frozen-attention-tiny` come from fe1b_sim1.

(2) R40-08.attention and R40-08.decode

```
R40_CONTRACT=v3 python3 tools/r40/v3_r40_08_gates.py [--dry-run] \
  --native $E/R40-08.attention/board_native_fe1b \
  --extra-board $E/R40-08.attention/board_native_fe1b/hbm14.jsonl.gz $E/R40-08.attention/board_native_fe1b/large24.jsonl.gz \
  --board $E/R40-02.zero-transfer/board_queue_fe1b \
  --model-report $E/R40-08.decode/board_model_fe1b/model_report.json.gz \
  --bucket-audit $E/R40-08.decode/board_model_fe1b/bucket_model_capability.json \
  --clock-evidence $E/R40-08.decode/board_model_fe1b/qualified_clock.json
```

Attention: 15/15 original attention/llama/GPT-2 graphs on the board (max rel. err 6.2e-4). The 4 HBM14 and
5 large24 attention-family points are also required to pass (`--extra-board` names both, where the default
names only HBM14). The 14 new SDPA cases and mode switching come from fe1b_sim1. The 64-step cache update is
the full-model board generation. fe1b_sim1 has no `bucket-decode-64` step, and the record says so
(`other_64_step_evidence_not_included`). Decode: 64 tokens equal the independent reference, one decode
compilation, 77 device-timed launches per step, on-board audit of this exact report (input sha256 `4658ef4b…`).

(3) R40-04.hbm, R40-04.robustness, R40-05.dispatch, R40-06.nuca, R40-09.suites

```
R40_CONTRACT=v3 python3 tools/r40/v3_memsys_gates.py [--dry-run] \
  --only R40-09.suites R40-04.hbm R40-04.robustness R40-05.dispatch R40-06.nuca \
  --ci-dir $E/ci_39281384427 \
  --board-run $E/R40-04.hbm/board_run_fe1b \
  --window-samples $E/R40-04.hbm/board_run_fe1b/window_samples.json \
  --model-report $E/R40-08.decode/board_model_fe1b/model_report.json.gz \
  --large24 $E/R40-04.hbm/board_run_fe1b/large24.jsonl \
  --hbm-model-fidelity uncalibrated-declared
```

`--gb10-dir` (`memsys_gb10/`) and `--semantics-dir` (`R40-06.semantics/`) keep their defaults, both fe1b.
`--ci-dir` replaces the default `ci_39183153944` (head 8820782, tree 80dd) with the fe1b CI run, so the CI
lanes are also "measured on the delivery tree". With the old 80dd CI run the result was the same
(all pass through the amendment-equivalence rule), but that run is not used. R40-06.semantics is left out
with `--only`; it was already capability-pass on the same GB10 and semantics evidence and was not re-recorded.
The HBM model fidelity is the declared sec 3.4 uncalibrated branch (decided earlier). The fixed-input
journal replay still has to pass, and it does. No `--preserve` was used: the inputs are judged where they are
committed.

(4) R40-01.device. `tools/r40/native_inventory.py` writes the v2 record that v3 accepts only when it
is bound to the delivery identity (see "Tooling changes").

```
python3 tools/r40/native_inventory.py --native $E/R40-04.hbm/board_run_fe1b/native \
  --software-tree $D --image-sha256 $I \
  --operating-clock-evidence $E/R40-08.decode/board_model_fe1b/qualified_clock.json [--evidence]
```

77/77 points plus one configuration plus the new identity check: every record on tree fe1b, image a88ebdf5,
90 MHz and clock evidence `8ad4b3bd…`.

(5) R40-05.multicore. `tools/r40/multicore_board_acceptance.py` (sha256 `4fed5854…`, unchanged) can only run on
the FPGA host. It hashes the xclbin, and `tnca.operating_clock.qualified_clock` re-reads the clock's raw
timing/route/CDC files at their absolute host paths. It also checks every compiler bundle at its absolute host
path (`/home/zzhouch/tnca_runs_r40_v3x2/bundles/...`). A local dry run (without `--evidence`, with the xclbin
fetched to scratch: sha256 a88ebdf5) refuses at the first of these, with
`FileNotFoundError: /home/zzhouch/tnca_build/r40_attempt03_conditional_90mhz/operating-clock-audit-v2/timing.rpt`.
The recollect_x2 queue ran it on the host as stage `physical-multicore-capability-audit`
(`board_audit_fe1b/command.json`, exit 0, `R40-05.multicore: capability-pass`, 94/94 checks). The host
wrote the v2 gate record `host_gate_evidence/R40-05.json`. As for earlier identities
(`evidence/r40/v2/convergence`), that record is imported unchanged into `sim/build/r40/v2/R40-05.json`
after an independent cross-check:

```
python3 $E/R40-05.multicore/board_audit_fe1b/crosscheck_host_audit.py [XCLBIN]     # 13/13 with the xclbin
python3 -c "import json; from pathlib import Path; \
  src=json.loads(Path('$E/R40-05.multicore/board_audit_fe1b/host_gate_evidence/R40-05.json').read_text())['gates']['R40-05.multicore']; \
  p=Path('sim/build/r40/v2/R40-05.json'); d=json.loads(p.read_text()) if p.exists() else {}; \
  d.setdefault('gates',{})['R40-05.multicore']=src; p.write_text(json.dumps(d,indent=1,default=str)+'\n')"
```

The cross-check verifies the following.
- The host auditor equals the repository file.
- Report, gate record, identity, v2 lock and clock agree.
- Every audit input (checkpoint, clock, native77) equals the committed copy.
- The queue stage, its exact command and its log are as recorded.
- Every file of the 77 native compiler bundles named by the records is on the host with the recorded hash (listing).
- The auditor's `checkpoint_checks` and `native_inventory.audit`, re-run locally, give the host's first 93 checks exactly (15 checkpoint and 78 inventory checks; the 94th is the in-place bundle/provenance loop).
- The xclbin hashes to the delivery image.

(6) R40-01.mixed-device

```
python3 tools/r40/v3_conformance_docs.py --run-dir $E/R40-02.zero-transfer/board_queue_fe1b \
  --stage complete-no-cuda-host-conformance --host-class no-cuda-host --backend fpga-u280-multicore \
  --launch-controller-sha256 309b1e9d0d33b36d7db80eb988280933e21ecf603e2395a7e654844890df013c \
  --live-source-tree $D --out $E/R40-01.mixed-device/no_cuda_host_fe1b.json
# not executed: the CUDA-host half failed (below). Once a complete fe1b CUDA-host run exists:
#   python3 tools/r40/v3_conformance_docs.py --run-dir <copy of ~/r40v3_cuda_fe1b> --stage cuda-device-conformance \
#     --host-class cuda-host --backend fpga-multicore-sim --launch-controller-sha256 <controller sha256> \
#     --live-source-tree $D --out $E/R40-01.mixed-device/cuda_host_fe1b.json
#   python3 tools/r40/device_conformance_acceptance.py --cuda $E/R40-01.mixed-device/cuda_host_fe1b.json \
#     --no-cuda $E/R40-01.mixed-device/no_cuda_host_fe1b.json --software-tree $D --image-sha256 $I \
#     --out $E/R40-01.mixed-device/audit_fe1b.json [--evidence]
```

The no-CUDA half passes on its own (`device_conformance_acceptance.audit(..., 'no-cuda-host')`: 16/16,
`mixed_devices: skipped: no CUDA device on this host`, image a88ebdf5, terminal same-source controller).

The CUDA half did not complete. The GB10 run `~/r40v3_cuda_fe1b` (controller sha256 `35c32b3b…`, source
`~/r40v3_capgate_fe1b`) re-hashes to tree fe1b on GB10. Its `device_tests.py` (`7c68246f…`) and the three
unit harnesses equal the repository copies. It ended `failed-preserved` at 2026-10-20T01:25Z:
- storage-owner-unit, signed-copy-range and native-recovery exited 0.
- cuda-device-conformance exited 1 with `r40-device-tests FAIL (15/16)`.

The failing case is `mixed_devices`, at its first CUDA statement `a = torch.randn(256, device='cuda')`. It
raised `torch.AcceleratorError: CUDA error: out of memory`; the driver logged
`Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain`. The CUDA context could not be
created, before any TNCA work in that case. `torch.cuda.is_available()` was true. The other 15 cases passed: eviction before it, and cache_hits,
checkpoint and the rest after it.

GB10 is a shared unified-memory host (119 GB; load average 9.8 observed during the run). Afterwards (01:26Z)
`nvidia-smi` showed no compute process on the GPU and 27 GB free. That points to transient memory pressure
from the shared host, not to the product, but it is not proven: dmesg is not readable for this user. The
80dd CUDA run (`cuda_host_80dd.json`) passed this case with the same `device_tests.py`.

As instructed, the run was not repeated. The raw run is preserved in `cuda_host_fe1b_failed/`.
A scratch dry run of `device_conformance_acceptance.py` on that run refuses it with
`terminal same-source controller required`; no `--evidence` record was written. The gate stays
not-measured, and `sim/build/r40/v2/R40-01.json` still holds the old 44051ad9 record, which acceptance does not take.

## Tooling changes (tools/r40, outside the product tree)

- `v3_r40_02_gates.py`: new `model_stage()`, used by `model_checks` (and by `v3_r40_08_gates.decode_checks`).
  The auditors looked for the model as a top-level queue stage `frozen-original-model`, which is the 80dd
  `tnca_v3_functional` layout. The bc5a/fe1b re-collection runs the model inside its
  `full-original-functional-delivery` stage (`tools/r40/conditional_delivery.py`), so the auditors reported
  "board model stage not complete" on a complete run. The top-level layout is judged exactly as before.
  For the nested layout the auditor requires all of the following.
  - The queue stage exited 0 and its command carries `--bucket`, the delivery image and the delivery tree.
  - The preserved `<stage>/delivery.json` has schema `tnca-conditional-delivery/1`, status
    `functional-stages-complete-review-required` and fixed-bucket policy.
  - Its `harness_sha256` equals the repository `conditional_delivery.py`.
  - Its identity is the delivery tree, image and 90 MHz.
  - Every nested stage ran in order and exited 0.
  - The nested `frozen-original-model` command wrote under the queue command's `--out`.

  The existing native_model.py/`--bucket`/image/tree argv checks then apply to the nested command.
  Negative tests are listed below.
- `v3_r40_08_gates.py`: `decode_checks` takes the model stage through `model_stage()`. For a top-level stage
  this also reads its command, which it did not before. The bucket-audit stage check is unchanged.
- `native_inventory.py`: optional `--software-tree`, `--image-sha256` and `--operating-clock-evidence`, which
  go together. When given, a new check requires every record to carry exactly that tree (provenance and
  platform), image (full SHA and `xclbin:` prefix), clock_hz, clock-evidence sha256 and embedded clock evidence.
  The record then names `identity`. Without this, `acceptance.py` reports the v2 record as
  "measured on None/None" and can never accept R40-01.device under v3. Without the new arguments the
  behaviour is unchanged (`make r40-device`, `multicore_board_acceptance`, `v3_memsys_gates` and
  `device_time_v2_tests` use `audit()`, which is untouched; `device_time_v2_tests` passes).

Negative tests of `model_stage` on mutated copies of `board_queue_fe1b` gave the following.
- `fail`: a nested stage exit 1, the nested model stage missing, the nested tree changed, the harness hash
  changed, the status not complete, the model `--out` elsewhere, or the outer stage exit 1.
- `not-measured`: `delivery.json` absent.
- 80dd `board_queue_80dd` still `not-measured` on its 80dd identity.

## Gate table (`make r40-v3-acceptance`)

Run 2026-10-20 ~01:28Z after the audits above (the full output also lists performance and deferred gates):

| Gate | Kind | Status |
|---|---|---|
| R40-00.protocol / b39-reproducible / frozen | capability | capability-pass |
| R40-01.device | capability | capability-pass |
| R40-01.mixed-device | capability | not-measured (CUDA-host half failed, see (6)) |
| R40-01.device-time | capability | fail (S_geo 1.0015, CI [0.9947, 1.0080]; see `R40-01.device-time/regression_diagnosis.md`) |
| R40-02.zero-transfer | capability | capability-pass |
| R40-02.resident | perf | perf-pass |
| R40-02.batched | perf | perf-pass |
| R40-02.stress | capability | capability-pass |
| R40-03.tiles, R40-03.contract | capability | capability-pass |
| R40-03.wide-blocks, R40-03.gemm-tile | perf | no-significant-gain |
| R40-04.hbm, R40-04.robustness | capability | capability-pass |
| R40-04.bandwidth | perf | perf-pass |
| R40-05.multicore, R40-05.dispatch | capability | capability-pass |
| R40-05.scaling | perf | perf-pass |
| R40-06.nuca, R40-06.semantics | capability | capability-pass |
| R40-06.streaming-scaling | perf | no-significant-gain |
| R40-06.locality | perf | deferred |
| R40-07.pipeline | capability | capability-pass |
| R40-07.long-k | perf | no-significant-gain |
| R40-07.matrix | perf | deferred |
| R40-07.policy | capability | not-measured |
| R40-08.attention, R40-08.decode | capability | capability-pass |
| R40-08.prefill | perf | no-significant-gain |
| R40-09.suites | capability | capability-pass |
| R40-09.model | perf | perf-pass |
| R40-09.autotune, R40-09.holdout | capability / perf | deferred |
| R40-10.acceptance | capability | not-measured |

Work packages passing: R40-00, R40-02, R40-04, R40-05, R40-09.
`R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM_V3: NOT PASSED`.

Of the eleven gates this page covers, ten are capability-pass; R40-01.mixed-device is not-measured.

## Update 2026-10-20: R40-01.mixed-device completed

The two failed GB10 CUDA-host attempts (`cuda_host_fe1b_failed/`, `cuda_host_fe1b_attempt3_failed/`) both stopped at CUDA context creation (CUDA_ERROR_OUT_OF_MEMORY in cuDevicePrimaryCtxRetain) before any test statement ran. The host was memory-exhausted by this session's own SystemC model runs (about 13 `tnca-chip` processes on HBM-size points), not by another user. Those runs were stopped and capped. Attempt 4 is the same controller (sha256 `35c32b3b…`, unchanged), started behind a `free >= 40 GB` guard (`cuda_host_fe1b_run/gate_passed_utc`). It completed with every stage exit 0 and the device tests 16/16, including `mixed_devices` (2 rounds of CUDA and tnca graphs in one process).

```
python3 tools/r40/v3_conformance_docs.py --run-dir $E/R40-01.mixed-device/cuda_host_fe1b_run --stage cuda-device-conformance \
  --host-class cuda-host --backend fpga-multicore-sim \
  --launch-controller-sha256 35c32b3bc0bd9d5c188478fd86c7a86dca5f9882c9e6bbb506047cd837dd0d4b \
  --live-source-tree $D --out $E/R40-01.mixed-device/cuda_host_fe1b.json
python3 tools/r40/device_conformance_acceptance.py --cuda $E/R40-01.mixed-device/cuda_host_fe1b.json \
  --no-cuda $E/R40-01.mixed-device/no_cuda_host_fe1b.json --software-tree $D --image-sha256 $I \
  --out $E/R40-01.mixed-device/audit_fe1b.json --evidence      # capability-pass, 32/32
```
