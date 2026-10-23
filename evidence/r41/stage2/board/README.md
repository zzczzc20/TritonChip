# R41 Stage 2 board batch: run-1 image at 100 MHz (prepared, not started)

This batch checks the full run-1 image ([`../run1/README.md`](../run1/README.md)) on the U280 against S2-01 and the board items of S2-04/S2-05 (`doc/r41_tasks_and_acceptance.md`). It covers the actual 100 MHz clock, with the image hash, source, parameters and clock bound together; the original full-suite numeric regression; FP8 on the board; the 8 GiB window; and a 2-hour continuous mixed run. **Nothing has been run and nothing has been loaded.** The user launches it with the commands below.

## Image

| Item | Value |
|---|---|
| xclbin | `~/tnca_images/r41s2_run1_c9f25787494a/tnca_u280.xclbin`, sha256 `30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de` |
| UUID (`xclbinutil --info`, read on acf3019 2026-10-23) | `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`; `DATA_CLK` 100 MHz |
| Source | `r41s2-t2` @ `c9f25787494a`, archive sha256 `4a2dcaf6…abfadd48` |
| Parameters | `evidence/r41/stage2/integration/image.json`, profile `core-vrf-sync-v5-x8`, `HAS_TENSOR_XPOSE=2`, 3 cores, 3 memory ports |
| Routed checkpoint | `level0_wrapper_routed.dcp`, sha256 `311bd49c…7045215` |
| HBM map | [`xclbin_hbm_map.json`](xclbin_hbm_map.json): a static parse of the xclbin (the card was not touched). Its group connectivity (`MBG[0:31]` on all three outlets), HBM topology and 4/8 GiB allocation are identical to B40's (`evidence/r41/hbm8/xclbin_hbm_map.json`) |

## Software tree: decision (a), merge

The branch `r41s2-board` is `r41s2-run1` (9f5587d) with `r41-sw` (56694cc, FP8 software) merged in, then `r41-hbm8` (2205c13, the 8 GiB cases). The batch runs this merged tree, and no other.

- **Why not (b).** `r41-sw` as-is carries the X8 RTL *without* the T2 fixes (`tnca_tensor.v`, x_qe), so its `rtl/` is not the image's. It also lacks the run-1 evidence and the integration image files, and it has no hbm8 tooling. A board record binds one `software_tree`, so running (b) would bind every record to RTL that is not on the card.
- **Image RTL unchanged.** After the merge, the tree hashes of `rtl/`, `tb/`, `flow/`, `xls/`, `spec/r1_profiles.json` and `compiler/targets` equalled those of `c9f2578`. The hbm8 merge then changed only testbench files under `tb/`. The one conflict, `rtl/core/tnca_tensor.v`, kept the image side, which is X8 plus the T2 `x_qe` fix. `doc/support_matrix.md` was regenerated (`gen_support_matrix.py --check` passes).
- **Compatibility.** The two branches have byte-identical copies of `compiler/targets/tnca-sim-vrf-v5-x8.json` (capability `tensor_transpose_8bit`), `spec/r1_profiles.json` (profile `core-vrf-sync-v5-x8`), `python/tnca/device.py` (x8 profiles), `compiler/python/tnca_cc/target.py`, the X8 SystemC model and the ISA text. The multicore backend takes `image_profile=core-vrf-sync-v5-x8` with `core_count=3` and `memory_ports=3`. Every fpga step sets `TNCA_PROFILE=core-vrf-sync-v5-x8`, so the compiler emits `t.ld.async.b.t`, which the image implements (`HAS_TENSOR_XPOSE=2`). The run-1 xclbin connects all three outlets to `MBG[0:31]`.
- **Recorded identity.** [`image_source.json`](image_source.json) lists the sha256 of all 128 image-source files from `c9f2578` (`files_sha256 14abdc9d…`). The `s2-identity/source` step re-hashes those files in the unpacked board tree and refuses any difference. Every board record also carries `archbench` `software_tree` of the board tree, and experiment.py refuses a record whose tree changed.
- **Software change.** `python/tnca/operating_clock.py` now accepts a second reviewed qualification. `r41s2-run1-100mhz` is bound to image `30e79769…` and to exactly 100 MHz. The R40 rule (≤ 91 MHz, any reviewed image) is unchanged.

## Clock qualification at load

The runtime refuses a board record unless `qualified_clock.json` passes `tnca.operating_clock.qualified_clock`, and that file can only be written after the image is loaded. [`clock_qualify.py`](../../../../tools/performance_harness/clock_qualify.py) derives every field from raw files:

1. Before waiting, the queue runs the post-route audit once on the routed checkpoint (`flow/fpga/audit_r40_operating_clock.tcl DCP OUT 100 placement`: Vivado analysis only, no card). It produces timing, route, CDC and actual core placement (SLR0/1/2) reports.
2. After the load, `capture` reads sysfs only: the UUID, three clock-counter samples and the ICAP readback. It refuses unless the UUID is `49964b17…` and every sample agrees.
3. `assemble` requires all of the following: a 10 ns analysis period; WNS, WHS and WPWS ≥ 0 with no failing endpoints; every routable net routed with 0 errors; three cores in their SLRs; a measured 100 MHz; and critical CDC findings exactly equal to the 56 platform findings reviewed for attempt 3. On the run-1 reports in `../run1/run1-reports.tgz` they are equal, 56 out of 56. Any difference means a new review is needed. The output is then re-read through `qualified_clock`.

The file is written to `~/tnca_build/r41s2_run1_c9f25787494a/r41s2-board/qualified_clock.json`, and every manifest names it. Its sha256 is bound into each record.

## Registered manifests (`spec/r41/stage2/board/`, launch order)

`tools/performance_harness/stage2_board.py --check` confirms that the registered manifests equal the generated ones. The batch manifests reuse the Stage 1 plan (`spec/r41/stage1/plan.json`, sha `6eb1a3d3…`) and its frozen suites; only the board block differs.

| Manifest | What | Board hours (est.) |
|---|---|---:|
| `s2-identity` | The image-source check; `clock_qualify.py verify` (counter now, xclbinutil UUID and `DATA_CLK`, qualified clock); 3 small ArchBench points with `clock_hz` 100 MHz in their records. **If this fails, the queue stops.** | 0.1 |
| `s2-anchors` | Stage 1's three anchors × 3 interleaved sessions | 0.12 |
| `s2-fp8-cases` | `fp8_cases.py`, full: casts, plumbing, `_scaled_mm` per-tensor/rowwise including e4m3×e5m2, linears. The summary requires every case to pass and `t.ld.async.b.t` (X8 transposed load) in the dumped kernels | ~1 |
| `s2-fp8-blockwise` | `fp8_blockwise_cases.py`, full: blockwise `_scaled_mm` and `_scaled_mm_v2` | ~0.5 |
| `s2-hbm8`, `s2-hbm8-big` | The hbm8 manifests with this board: window, negatives, 34 samples, alias, crossing, DMA and clamp; then bigoffset and stream | 0.42 |
| `s2-original-suite` | The original full suite (113 points, `hbm.json`), numeric references, no fallback | 2.05 |
| `s2-fp8-hub` | `fp8_kernels_cases.py`: Hub `finegrained-fp8` v4, compile and transformers 5.20.2 `FP8Linear`; then v1 | ~1 |
| `s2-qwen-fp8` | `qwen_tnca_layers.py`: Qwen3-VL-2B-FP8 layer 13's 7 FP8 linears (M = 81 and 1), and the decode pair 13–14 (bucket 32, 4 steps) | ~0.7 |
| `s2-endurance` | `endurance_mixed.py`: ≥ 2 h of continuous rounds over 8 ArchBench points (HBM add, GEMM, HBM decode, attention, softmax, conv, GPT-2 block, layernorm), plus `fp8_cases --quick --group scaled_mm` every third round. Each session records its wait4 peak RSS. Host samples every 60 s (MemAvailable, own RSS and fds, the card probe). Peak-RSS and cycle slopes are taken per configuration. The run stops on any card change | 2.1 |

Total: about 8–9 board hours.

The FP8 Hub and Qwen manifests need inputs that acf3019 lacks: transformers, kernels, the HF snapshots and the CPU reference. `stage_inputs.sh` copies the snapshots, `ref_fp8` and the Hub main checkout from GB10. The queue builds `~/r41s2_fp8env` before it waits: a venv over `tc_env/torchenv` with transformers 5.20.2, kernels 0.17.2, accelerate 1.15.0, huggingface_hub 1.33.0 and safetensors 0.8.0, the same versions as GB10's `flux-venv`. It checks for torch 2.14.0 and triton 3.8.0. If either step is missing, those two manifests are skipped and logged, and the rest still run.

## Launch (the user)

Run these three commands in order, on three machines. `C` must be the same commit in every one.

```sh
# 1. local: the source archive of the board branch, to GB10
cd /home/ubuntu/TritonChip-r41s2-board && git fetch origin && git checkout --detach origin/r41s2-board
C=$(git rev-parse --short=12 HEAD); git archive --format=tar.gz -o /tmp/r41s2_board_$C.tar.gz $C
scp /tmp/r41s2_board_$C.tar.gz zhongchun@143.89.78.224:

# 2. GB10: stage archive, queue script and FP8 inputs on acf3019 (files only)
tar xzf ~/r41s2_board_$C.tar.gz -O ./evidence/r41/stage2/board/launches/stage_inputs.sh > ~/r41s2_stage_inputs.sh
bash ~/r41s2_stage_inputs.sh $C ~/r41s2_board_$C.tar.gz

# 3. acf3019: start the waiting queue
setsid nohup bash ~/r41s2_board_queue.sh $C > ~/r41s2_board_$C.nohup 2>&1 < /dev/null &
```

Progress: `~/r41s2_board_$C/queue.log` (it ends `S2-BOARD-QUEUE-DONE`). Per-manifest logs: `~/r41s2_board_$C/<manifest>.log`. Reports: `~/r41s2_board_$C/out/<manifest>/report.json`. To rerun after a stop, run the same command again. It takes a lock, does not load again when the image is already on the card, does not repeat the audit or the clock capture, and experiment.py reuses completed steps.

Queue exits: 2 = lock held; 3 = an archive, image, xclbin UUID/clock, checkpoint or audit failure (nothing loaded); 4 = program.sh refused, or the card was busy at load; 5 = the UUID or 100 MHz did not appear after the load, or the clock qualification was refused; 6 = the card changed or was busy between manifests; 7 = `s2-identity` failed.

## Dry runs (local, no card)

`python3 -m unittest test_stage2_board` (in `tools/performance_harness`): **14/14 pass**.

- Manifests: registered equals generated. Every board block is image `30e79769`, UUID `49964b17`, 100 MHz and profile x8, and every fpga step's environment names the run-1 xclbin and the qualified clock.
- All 10 manifests go through experiment.py with a fake image (UUID embedded), fake clock evidence and stubbed fpga commands. The report steps run for real: the source check passes on the tree, and the FP8 summary finds the X8 load. All 10 end `completed-experiment`. Probes of 90 MHz, the B40 UUID and a busy card are each refused before any command runs.
- `clock_qualify`: capture on a fake sysfs, then assemble on the real run-1 timing, route and CDC reports. Results: WNS 0.016, WHS 0.003, 56/56 CDC equal to the review. The 100 MHz review refuses another image. A missing CDC row, a 90 MHz capture and a wrong UUID are each refused.
- `endurance_mixed`: rounds with the wait4 RSS shim and FP8 sessions; a card change stops the run and the result is `fail`.
- `board_queue.sh` on a fake home and sysfs:
  - load (`--check`, `--go`, `--status`), then 100 MHz, then manifests, then DONE;
  - a rerun does not load again;
  - 90 MHz after the load exits 5 with no manifests run;
  - a non-critical failure continues; an `s2-identity` failure exits 7;
  - a busy card, or a Stage 1 queue that is still running, never reaches program.sh.
- `board_batch.py --dry-run`: identity 3 runs (0.014 h), anchors 9 (0.115 h), original 113 (2.048 h), endurance mix 8 per round (0.063 h per round).
- Existing tests: test_experiment, test_board_tools, test_hbm8, test_stage1_acceptance, operating_clock_tests: OK. `hbm8_manifest.py --check`: OK.

Simulator twin on GB10 (the merged tree f30459f; `systemc-tlm-grid` with `TNCA_PROFILE=core-vrf-sync-v5-x8` and an 8 GiB window): `fp8_cases.py --quick --group scaled_mm --group linear` passed **8/8** bitwise. The `fp8-summary --require-x8` step then found `t.ld.async.b.t` in 3 dumped FP8 dot kernels and returned `pass`. That run exposed a bug: the summary read only `cases`, while `fp8_cases` writes `rows`. The bug is fixed, and the summary now reads both.

What has not been exercised: any FP8, Hub or Qwen tool on `fpga-u280-multicore` (the simulators passed them, `evidence/r41/fp8`); the Vivado audit on the run-1 checkpoint (the same tcl ran on attempt 3); and the FP8 venv build on acf3019.
