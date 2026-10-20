# R40 v3 delivery evidence

Contract: `doc/r40_tasks_and_acceptance_v3.md`, `doc/r40_v3/plan.json`, locked by
`doc/r40_v3/lock.json` (`tools/r40/contract_v3.py`, `make r40-v3-contract-check`).
Judged by `make r40-v3-acceptance`.

Delivery identity:

| | |
|---|---|
| Image | `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` (3 cores SLR0/1/2, 3 outlets) |
| Clock | 90 MHz measured (90 MHz WNS +0.016 ns; 100 MHz not claimed) |
| HBM window | 4 GiB (PC0..15) |
| Product tree | `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75` (`doc/r40_v3/identity_amendment.json`; was `files:80dd1a82…`, then `files:bc5a778e…`) |

## Board work

| Run | Host dir | Identity | Purpose |
|---|---|---|---|
| `launches/functional_v3_launch.json` | `~/tnca_v3_functional` | 80dd | Model, conformance, large24, graphs24 after the `native_model.py` harness fix; early correctness signal (superseded) |
| `launches/campaign_launch.json` | `~/tnca_v3_campaign/run` | bc5a | All v3 performance experiments (`tools/r40/v3_campaign.py`) |
| `launches/recollect_launch.json` | `~/tnca_v3_recollect`, `~/tnca_runs_r40_v3x` | bc5a | Complete functional re-collection: window/PC probes, checkpoint, native77, HBM14, fixed-bucket model, audits, no-CUDA conformance, large24, graphs24 |
| `launches/cuda_bc5a_launch.json` | GB10 `~/r40v3_cuda_bc5a` | bc5a | CUDA-host conformance and recovery for `R40-01.mixed-device` |
| `launches/x2_launch.json` (`recollect_x2_controller.py`) | `~/tnca_v3_recollect_x2`, `~/tnca_runs_r40_v3x2` (source `~/tnca_src_r40_v3x2`) | fe1b | Complete functional re-collection on the delivery tree (same stages as bc5a); all 30 stages exit 0, `complete-review-required`. Judged in `capability_fe1b.md` |
| `launches/cuda_fe1b_launch.json` | GB10 `~/r40v3_cuda_fe1b` (source `~/r40v3_capgate_fe1b`) | fe1b | CUDA-host conformance and recovery for `R40-01.mixed-device`; `attempt 2 `failed-preserved`: cuda-device-conformance 15/16 (`mixed_devices` hit CUDA_ERROR_OUT_OF_MEMORY in cuDevicePrimaryCtxRetain on the shared unified-memory host, before any test statement ran); raw copy `R40-01.mixed-device/cuda_host_fe1b_failed/`, kept on GB10 as `~/r40v3_cuda_fe1b_attempt2_cudaoom`. Attempt 3: the same controller (sha256 unchanged) relaunched 2026-10-20 01:29 UTC behind a free-memory guard (>= 16 GB free) |

## Compiler evidence

- `identity_amendment/`: the scan/sort narrow-block fix, with the 112-compilation
  before/after identity diff.
- `compile_preflight/`: the pre-declared candidate rule for wide blocks and
  GEMM tiles (offline compile coverage, no performance input).

Gate status is the output of `make r40-v3-acceptance`; this page is updated as
evidence lands.
