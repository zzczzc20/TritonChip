# R40 report: native PyTorch device, three physical cores, NUCA and HBM on the U280

Contract: `doc/r40_tasks_and_acceptance_v3.md` (v3, the one-time relaxation of
v2 for delivery on the loaded 90 MHz image; it inherits v2 and v1). Sign-off
name `R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM_V3`. Machine judgement:
`make r40-v3-acceptance`.

> Status: IN PROGRESS. Sections marked *pending* are filled from the board
> results as they land; nothing here is asserted before its evidence exists.

## 1. What was delivered

| | |
|---|---|
| Board | Alveo U280 (`xcu280-fsvh2892-2L-e`), XRT 2.16, Vitis/Vivado 2023.2 |
| Image | `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` (`tnca_u280.xclbin`, UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`) |
| Cores | 3 physical TNCA cores (core-vrf-sync-v5), one per SLR (SLR0/1/2), 3 memory outlets, TMU per SLR |
| Memory | 4 GiB exposed HBM window over PC0..15; PC16..31 not exposed and not certified |
| Clock | DATA_CLK measured 90 MHz on the loaded card (MMIO counter samples) |
| Software | product tree `files:bc5a778ebc5f4afebf6c1a9fbf3b2d8717cb02e118134ebbb73fb5a927fc8183` (`doc/r40_v3/identity_amendment.json`) |

### 1.1 Frequency exception (authorized)

The first-round target was 100 MHz. The routed 100 MHz implementation fails
setup (WNS −1.002 ns, TNS −6679 ns; the worst cone is the memory-active-port
control to the LSU pending logic, 14 logic levels, 11.18 ns data delay of which
10.31 ns routing). Vitis placed DATA_CLK at 90 MHz; at 90 MHz the same routed
design meets every constraint:

| WNS | TNS | WHS | THS | WPWS | Failing endpoints | Route errors | Routed nets |
|---|---|---|---|---|---|---|---|
| +0.016 ns | 0 | 0.000 ns | 0 | 0.000 ns | 0 | 0 | 1,160,009 |

The user authorized the ~10 MHz reduction (2026-10-18). R40 makes no 100 MHz
claim; frequency work moves to the next implementation together with every
item that needs a new bitstream.

### 1.2 Resources (placed, per SLR)

| Resource | SLR0 | SLR1 | SLR2 | SLR0 % | SLR1 % | SLR2 % |
|---|---|---|---|---|---|---|
| CLB LUTs | 322,934 | 325,056 | 281,718 | 73.45 | 75.24 | 65.21 |
| CLB registers | 159,717 | 144,350 | 105,429 | 18.16 | 16.71 | 12.20 |
| Block RAM tiles | 139 | 153.5 | 121.5 | 20.68 | 22.84 | 18.08 |
| URAM | 0 | 0 | 0 | 0 | 0 | 0 |
| DSPs | 171 | 189 | 175 | 5.94 | 6.15 | 5.70 |

SLR crossings: SLR1↔SLR0 14,224 of 23,040 (61.7 %), SLR2↔SLR1 8,071 (35.0 %).
CDC: the 56 Critical findings in the vendor platform structures were reviewed
by exact endpoint; no TNCA endpoint appears in the CDC crossing report.
Raw reports: `evidence/r40/v2/convergence/attempt03/qualified90_raw.tar.gz`,
`slr_util_placed_20261018.rpt.gz`, `operating_timing_90mhz_pass.json`.

## 2. Contract changes (v3)

See `doc/r40_tasks_and_acceptance_v3.md` for the text. In short:

1. All gates are judged on the image above at 90 MHz.
2. Performance gates pass on a significant gain: the 95 % paired bootstrap
   lower bound of S_geo above 1 on the complete frozen subset, every point
   correct. The v2 threshold is reported as a target, met or missed.
3. Board device-cycle sampling: a cold first call (never pooled), then one
   qualification sample, then three formal samples where the call is at most
   1 s. Device cycles reproduce within 0.2 % across runs.
4. Moved out of R40: `R40-07.matrix` (needs new MMA RTL; with the next
   frequency implementation), `R40-06.locality` (DCO/locality software path),
   `R40-09.autotune` and `R40-09.holdout` (TNCA_TUNE) — the latter three to R41
   by the user's decision.

## 3. Gate results

*pending* — the table is generated from `make r40-v3-acceptance` once the
board campaign and the functional re-collection finish.

## 4. Performance against the v2 targets

*pending* — from `tools/r40/v3_campaign.py judge`.

## 5. Problems found and fixed during delivery

| Problem | Effect | Fix |
|---|---|---|
| Large-model weights overlapped the 4 KiB TMU MMIO page at `0x7fff0000` | original large Llama logits out of tolerance (0.114) | allocator excludes the reserved page (1a7c643); error 0.0027 |
| Probe padding crossed the reserved page | HBM window probe assertion | fillers split at the page (0958db2) |
| `native_model.py` asserted on `warm` after deleting it | final-identity model stage stopped before step 0 | assert moved to the warm path (8f8126e) |
| Wide register blocks have no `tt.scan` lowering | `TNCA_MAX_BLOCK` > 16 refused `op.cumsum` | scan/sort kernels keep 16 lanes (b450904); all other artifacts byte-identical |

## 6. GPU comparison

*pending* — `evidence/r40/v3/gpu_comparison/`.

## 7. Reproduction

*pending* — `tools/r40/v3_reproduce.sh` from a clean archive of the delivery
commit, run as an independent session.
