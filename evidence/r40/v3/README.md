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
| Product tree | `files:bc5a778ebc5f4afebf6c1a9fbf3b2d8717cb02e118134ebbb73fb5a927fc8183` (`doc/r40_v3/identity_amendment.json`; was `files:80dd1a82…`) |

## Board work

| Run | Host dir | Identity | Purpose |
|---|---|---|---|
| `launches/functional_v3_launch.json` | `~/tnca_v3_functional` | 80dd | Model, conformance, large24, graphs24 after the `native_model.py` harness fix; early correctness signal (superseded) |
| `launches/campaign_launch.json` | `~/tnca_v3_campaign/run` | bc5a | All v3 performance experiments (`tools/r40/v3_campaign.py`) |
| `launches/recollect_launch.json` | `~/tnca_v3_recollect`, `~/tnca_runs_r40_v3x` | bc5a | Complete functional re-collection: window/PC probes, checkpoint, native77, HBM14, fixed-bucket model, audits, no-CUDA conformance, large24, graphs24 |
| `launches/cuda_bc5a_launch.json` | GB10 `~/r40v3_cuda_bc5a` | bc5a | CUDA-host conformance and recovery for `R40-01.mixed-device` |

## Compiler evidence

- `identity_amendment/`: the scan/sort narrow-block fix, with the 112-compilation
  before/after identity diff.
- `compile_preflight/`: the pre-declared candidate rule for wide blocks and
  GEMM tiles (offline compile coverage, no performance input).

Gate status is the output of `make r40-v3-acceptance`; this page is updated as
evidence lands.
