# R41 Stage 2 board results (run-1 image, 100 MHz)

Board batch: commit `86a3d929cf3d` (branch `r41s2-board`), run by `board_queue.sh` on the FPGA host acf3019 on 2026-10-25/26 (UTC).
Image: `r41s2_run1_c9f25787494a/tnca_u280.xclbin`, sha256 `30e79769...4e5de`, UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`, profile `core-vrf-sync-v5-x8`.
Qualified clock evidence: sha256 `49868f45...f0d7`. Software tree of the batch: `files:c560dca3...79290e`.
Outputs on the host: `~/r41s2_board_86a3d929cf3d/out/<manifest>/`; queue log `~/r41s2_board_86a3d929cf3d/queue.log`.

An `experiment.py` exit code of 0 only means that the commands finished. Everything below was taken from the per-point records, the step reports and the card probes, not from the exit codes. The `result`, `residual` and `conclusion` fields of the `experiment.py` reports of the passing manifests are all `not-recorded`; the verdicts here are this document's own.

## Summary

| Manifest | Verdict | Basis |
| --- | --- | --- |
| `s2-identity` | Passed | 3 of 3 points pass; clock counter 100 MHz; UUID and image hash as registered |
| `s2-anchors` | Passed | 9 of 9 sessions pass numerically; cycles within 0.01-0.2 % across 3 sessions; device time 8.3-10.4 % shorter than at 90 MHz |
| `s2-fp8-cases` | Passed | 100 of 100 cases on the board (49 `_scaled_mm`, 27 casts, 14 plumbing, 10 linear), 42 X8 kernels with `t.ld.async.b.t`; bit-exact except NaN sign encoding in 6 plumbing comparisons |
| `s2-fp8-blockwise` | Passed | 16 of 16 cases, 0 bit mismatches, every kernel carries the e4m3 MMA |
| `s2-hbm8` | Passed | 7 of 7 cases, 159 checks, 8 GiB window, PC0-PC31 covered, addresses up to 0x1fff10000 |
| `s2-hbm8-big` | Passed | 2 of 2 cases, 14 checks, single storage above 2 GiB, offset beyond int32 |
| `s2-original-suite` | Passed | 113 of 113 points pass (202 runs, none failed); 1.095x in device time against 96 matched Stage 1 points |
| `s2-endurance` | **Passed on second attempt; first failed (harness fault)** | After harness fixes, 250/250 sessions pass in 2.0611 h; first attempt retained below |
| `s2-fp8-hub` | Not run | the FP8 virtual environment did not build |
| `s2-qwen-fp8` | Not run | the FP8 virtual environment did not build |

## Clock and image identity

- Before every run the batch driver probes the card read-only. Every probe recorded in `s2-anchors` (10) and `s2-original-suite` (203), and the identity capture, shows UUID `49964b17-...` and clock counter `[100, 500, 450, 0]`. The endurance run's 123 host samples show the same. Every accepted original-suite record carries `clock_hz` 100000000, image hash `30e79769`, and `evidence_level` `measured-board`.
- The qualification (`s2-identity/clock.json`) agrees: `xclbinutil` reports DATA_CLK 100 MHz, the sysfs counter reads 100 MHz three times, the ICAP read-back says 100 MHz, status `pass`.
- Limits of this evidence: the counter has a resolution of 1 MHz, so it shows 99.5-100.5 MHz, not an exact 100.000 MHz. It is a measurement of the clock the image runs at, not a measurement per run. The cycle ratios below (compute-bound points gain 9-11 % in time at near-constant cycles) are consistent with 100 MHz.
- `s2-fp8-cases` and `s2-fp8-blockwise` write the stale label `"evidence": "rtl-simulation"` into their reports. The same reports state `backend: fpga-u280-multicore`, the image path and `hbm_bytes` 8 GiB, and every launch has device cycles, so the runs were on the board. The label is a harness defect to be corrected.

## s2-endurance: root cause

Every one of the 250 sessions in `mix/sessions.jsonl` fails with exactly `['qualified clock evidence identity']` (ArchBench points) or `['exit 1', 'no report']` (FP8 sessions). Two independent harness bugs:

1. **Unbound batch.** `Batch.check` compares the record's `operating_clock_evidence_sha256` with `self.state['clock_evidence_sha256']`. That entry, together with the image hash, clock and software tree, is written only by `Batch.execute()`. `endurance_mixed.run` builds a `Batch` and calls `run_one` directly, so the entry stayed `None` and every record was rejected. The other manifests go through `execute()` and are not affected. The records themselves are correct: all 240 pass the other checks (status `pass`, no fallbacks, tolerance, image `30e79769`, clock 100 MHz, knobs, launch map, memory window, sampling), and their clock evidence hash is the registered one. Because the batch marks a rejected run as failed, `serial_grid_cycles` and `device_seconds` are also `null` in the session rows (the raw records keep them). The unit test used a `FakeBatch` that never checked the identity, which hid the bug.
2. **FP8 output directory.** `run_fp8` created the output directory to hold its log, and `fp8_cases.py` refuses an existing `--out` (`FileExistsError`). In addition `run_fp8` read `cases` from the report while `fp8_cases.py` writes `rows` (the same kind of mismatch as the one fixed earlier in `hbm8_summary`). Also fixed.

The card was not at fault. Evidence from the failed run that remains useful but is **not accepted** as the endurance result: 240 ArchBench sessions over 2.03 h (30 rounds, 8 configurations x 30), all numerically within tolerance; per-configuration peak RSS flat (fitted growth +0.0002 to +0.0011 of the first value over the run; the -0.13 of `attention_prefill` comes from a 1.6 GB first session, a cold start, against 0.58 GB afterwards); device cycles per configuration within 0.04-2.5 % (max-min over median) with no trend (first-5 against last-5 means differ by under 0.5 %, except `conv2d` 0.6 %); no card change in 123 host samples.

## s2-anchors (3 sessions each, interleaved)

| Anchor | Stage 1 B40, 90 MHz: cycles / time | Stage 2, 100 MHz: cycles / time | Cycles | Time |
| --- | --- | --- | --- | --- |
| `op.add/hbm/fp16-n67108864` | 243,997,720 / 2.7111 s | 250,373,000 / 2.5037 s | +2.6 % | 1.083x |
| `op.matmul/medium/fp16-m128-k192-n128` | 231,745 / 2.575 ms | 235,416 / 2.354 ms | +1.6 % | 1.094x |
| `model.llama_decode/hbm/fp16-b1-p512` (serial grid) | 170,603,213 / 1.8956 s | 171,650,844 / 1.7165 s | +0.6 % | 1.104x |
| same, graph makespan | 204,863,817 / 2.2763 s | 210,557,215 / 2.1056 s | +2.8 % | 1.081x |

The ideal gain of the clock alone is 1.111x. HBM-bound work gains less because the memory clock does not move, so it costs more cycles. Spread over the 3 sessions at 100 MHz: add 0.02 %, matmul 0.2 %, decode 0.01 % (serial grid) and 0.22 % (makespan). Wall time per session is unchanged within noise (decode 62-64 s at both clocks; add 64 s against 60 s, with one 94 s first session at 100 MHz and one 113 s at 90 MHz), because compilation and host work dominate.

## FP8 on the board

- `s2-fp8-cases`: 100 rows, all `pass`, no host fallback, every graph resident, `t.ld.async.b.t` (X8 transposed load) in 42 dumped kernels. The gate of each case is a bit comparison of the device result with the same-order torch reference. Every primary comparison shows 0 mismatches except 6 plumbing comparisons (clamp-then-cast and dequantise), where the only mismatches are NaN sign encodings (for example `0x7f` against `0xff`), with 0 value mismatches. These cases are accepted by the harness's `nan_ok` rule. Informational comparisons with eager CPU torch show 1 ulp differences (for example 3 of 12,288 elements in one W8A8 linear); they are not a gate.
- Duration: 105 s in the step (67.5 s of case time), not the registered 1 h. The README estimate was carried over from the RTL simulator; the board runs the compiled kernels in seconds, and the 100 cases are small. The count (100) and the content of the report show that no case was skipped.
- `s2-fp8-blockwise`: 16 rows, all `pass`, 0 bit mismatches, the e4m3 MMA counter is 32 in every case; 440 s.
- Not run: `s2-fp8-hub`, `s2-qwen-fp8`. The queue's FP8 environment build failed (`fp8env.log`): the venv was created with `--system-site-packages` over `tc_env/torchenv/bin/python`, pip then downloaded `torch-2.14.1` (a 555 MB wheel) because it did not see the installed torch 2.14.0, and the version assertion `torch.__version__.startswith('2.14.0')` failed. The builder needs the existing torch made visible to the venv (for example with a `.pth` file or by installing with `--no-deps`) before this works.

## s2-hbm8 and s2-hbm8-big

All 9 cases `functional-pass` on one software tree, `mode board`, window base 0x0, size 8,589,934,592 (8 GiB), local limit 0x20000, AXI status clear at the end of each case.

- `window` (11 checks): window geometry and register read-back, `pc0-pc31-contiguous` (PC 0-31), XRT group buffer `MBG[0:31]` of 8 GiB, heap ends at the window, TMU page reserved.
- `samples` (76 checks): 34 probes placed and viewed at the intended addresses from 0x20000 to 0x20000fe00, `pc0-pc31-covered`, 31 PC boundary crossings, `above-4GiB-and-near-8GiB`, three cores in every launch (issued equals completed), three outlets with traffic.
- `alias` (36), `crossing` (17), `dma` (6; 256 MiB buffers at 0x180020000 and 0x1eff20000), `negatives` (10), `clamp` (3; a 16 GiB request is exposed as the 8 GiB group): device store and load at address bits 30, 31, 32 and 32+31, at virtual 4 GiB (0xffff8000), physical 4 GiB (0x100018000) and near the window end (0x1fff10000), all correct.
- `hbm8-big`: `bigoffset` (6 checks: reads and writes beyond 2 GiB, an offset exceeding int32, at 0x7fff1000) and `stream` (8 checks: a single storage above 2 GiB read exactly, `max_abs` 0.0, 37.8 M outlet read lines).
- `window` and `negatives` launch no kernel (host and MMIO checks); the launch cases cover all three cores.
- Not covered by a launch: bandwidth or timing (`performance_gate` is `not-measured`).

## s2-original-suite

113 of 113 points pass: 202 runs (113 qualification, 89 with three timed samples), none failed, no fallback, every record `measured-board` at 100 MHz on image `30e79769`, window 0 / 4 GiB / local limit 131072. The largest error against tolerance is 5.9 % of tolerance (`llama_prefill/large`, 2.93e-3 against 0.05). Board wall time 1.908 h.

Comparison with the Stage 1 B40 records (90 MHz, image `a88ebdf5`, `~/r41s1_board_25094197f6e6/out/board-original-suite`). The Stage 1 run was interrupted (the card was reloaded by another job, the state is `running` at 177 runs), so only **96 of the 113 points** have a Stage 1 value; the 17 large points (attention, bmm, conv2d, embedding, matmul 2048, and others) have none. The Stage 1 `attention_decode/large` run failed because the card no longer held its image. The comparison below uses the median device sample of the formal run of each point.

| Measure (96 matched points) | Stage 1 (90 MHz) | Stage 2 (100 MHz) | Ratio |
| --- | --- | --- | --- |
| Sum of device time | 842.1 s | 768.6 s | 1.096x (8.7 % shorter) |
| Sum of device cycles | 75.79 G | 76.86 G | +1.4 % |
| Per-point device-time speedup, geometric mean | | | 1.099x (min 1.005, max 1.215) |
| Per-point cycle ratio, geometric mean | | | 1.011 (0.914-1.106) |
| Sum of host wall time (qualification runs) | 4724 s | 4536 s | 1.04x |
| Per-point wall ratio, geometric mean | | | 1.00 (0.86-1.14) |

- Time gain is close to the 1.111x of the clock for compute-bound work: `llama_prefill/large` 1.095x (637.9 s to 582.6 s), `llama_prefill/hbm` 1.100x, `matmul 4096^3` 1.095x, `llama_decode/large` 1.111x, `mlp/large` 1.113x. The 16 points that take 1 s or more at 90 MHz gain 1.005-1.118x (median about 1.10x).
- Smaller gains: `rmsnorm` (median 1.017x; `rmsnorm/hbm` 1.005x, cycles +10.6 %), `layernorm` 1.053x, `transpose` 1.063x (`transpose/hbm/fp32` 1.031x, cycles +7.8 %), `gelu_tanh` 1.079x, `add` 1.083x. These are limited by memory or by cycles that grow with the clock.
- The one point with fewer cycles (0.914x) is `op.argmax/small` (33,477 against 30,606 cycles); at this size one sample is within the noise seen in the repeated tiny points (up to 37 % between samples of `transpose/tiny`). Single-run tiny points should not be read as an effect.
- Wall time is dominated by compilation, transfers and the host; the clock does not change it, so the wall speedup is 1.04x in total and 1.00x per point. The device-time speedup is 1.096x.
- The 17 points without Stage 1 data are not compared. A rerun of the Stage 1 suite at 90 MHz is planned (`r41_followup_chain.sh`); the table should be refreshed from it.

## Open items

1. `s2-endurance`: completed on second attempt after the harness fixes; see the update below. The first failed attempt remains preserved.
2. `s2-fp8-hub` and `s2-qwen-fp8` never ran; the FP8 virtual environment has to be rebuilt first.
3. The stale `evidence: rtl-simulation` label in the FP8 case reports, and the unrecorded `result`/`residual`/`conclusion` fields of the `experiment.py` reports, remain to be fixed.
4. Two tests in `test_stage2_board.py` fail on `r41-main` before and after this change: `test_every_manifest_completes_on_the_fake_board` (the `s2-identity` step fails on the fake board in this environment) and `test_image_source_matches_tree` (11 files differ from the recorded image source). They were not investigated here.

## Endurance completion update, 2026-10-26 UTC

The second attempt, commit `9bc8990`, ran on the same 100 MHz image from
06:12:27 to 08:16:07 UTC (2.0611 h), and passed 250/250 sessions (240 ArchBench,
10 FP8). Software tree: `files:a86af4ae352dd47d9910b35fbe411fcaef920c483bd6dcc7db330972584e1303`.
The fixes bind the batch identity before direct `run_one` calls, avoid creating
the FP8 output directory prematurely, and consume the report's `rows` field.
The first attempt's harness failure is not erased or counted as a pass.

The read-only archive `endurance-rerun.tgz` retains `report.json`,
`mix/endurance.json`, `mix/sessions.jsonl` and `mix/host.jsonl` from
`acf3019:/home/zzhouch/r41s2_endurance_9bc8990/out/s2-endurance/`.
`endurance-rerun-index.json` records archive and member SHA256 values.
Original-suite and endurance runs expose 4 GiB; the separate window cases
establish 8 GiB functionality. This endurance result does not establish bandwidth
or full Qwen inference.

Stage 1 now has 113 completed original points; the 96-point performance comparison
above remains the historical partial comparison and has not been recomputed here.
