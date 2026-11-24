# xl-cand2: the 32 xl points on the R41 run-3 image, candidate software, against xl-base (board)

Domain: board, U280, image `791b8af2...` (UUID `ea1fa1af-...`), 100 MHz, `device-cycles-serial-grid`, run-3 knobs, default compiler settings. Same loaded image as `xl-base` (no second load: `already-loaded`).
Registration: [`spec/r42/board/xl-cand2/`](../../../../spec/r42/board/xl-cand2/) (commit `da975c18`, pushed before the run). Software: branch `R42-xl-cand` `33b41987` = R42 head `9c795da3` merged with `R42-xl`
(`doc/r42_plan.md` conflict resolved keeping both sides); tree `files:67e7c6e9...`, the same tree as the earlier registration `xl-cand` (R42 head `ba5f20ef` + R42-xl, `aa558cdd`), which was registered and never launched.
Staged as `~/r42/src_xlcand2` on the FPGA host `acf3019`; queue directory `~/r42/q/xl-cand2`. Baseline: [`../xl-base/`](../xl-base/) (main `4b61abd8` + archbench xl additions, tree `files:64bf16ee...`).

Run (UTC): runner launched 2026-11-23 22:25:51 (`already-loaded`), identity 22:40:56-22:42:35 (clock pass, 3/3 points pass), xl suite 22:57:37 - 00:24:23 on 2026-11-24 (queue-end rc 0). 48 runs (32 qualification + 16 three-sample), 0 failed, 5,205 s process wall time.
Records [`records.tgz`](records.tgz) sha256 `c9663feb...`, logs [`logs.tgz`](logs.tgz), comparison [`comparison.json`](comparison.json)
(`python3 tools/r42/board_compare_xl.py --old ../xl-base/records.tgz --new records.tgz`).

**CERTIFIED**: all 32 points pass (numerics inside the frozen tolerance, no fallback) on both sides, both batches complete, same image and clock.
Geometric mean new/old over the 32 points **0.7688** (speedup 1.301x; median 0.9944, min 0.2609, max 1.0368);
sum of device cycles 149818089845 -> 97371712415 (0.6499).
By kind: 15 FP8 kernel points (`fp8_linear` 13, `fp8_act_quant` 2) 0.9993; 5 FP8 composites (layers, decoders) 0.6848; 12 fp16 points 0.5814.
Slower by more than 2 %: `op.linear/xl/fp16-m2048-k2048-n6144-bias0` 1.0368 (one sample on each side); every other point is within +0.6 %.
Largest error / tolerance on the candidate 0.893 (`fp8_act_quant`, as on the baseline); largest heap peak 2272010496 bytes on both sides.

| family | points | geomean new/old | old cycles | new cycles |
|---|--:|--:|--:|--:|
| argmax | 1 | 0.5924 | 4614775 | 2733845 |
| attention_decode | 3 | 0.6693 | 202229923 | 134636263 |
| attention_prefill | 2 | 0.2618 | 12210199742 | 3189797922 |
| embedding | 1 | 1.0000 | 74462105 | 74460826 |
| fp8_act_quant | 2 | 1.0000 | 26377697 | 26377355 |
| fp8_decoder_decode | 1 | 0.6947 | 2485589764 | 1726718215 |
| fp8_decoder_prefill | 2 | 0.6524 | 123675576420 | 81623787247 |
| fp8_layer_decode | 1 | 0.6543 | 78866886 | 51604507 |
| fp8_layer_prefill | 1 | 0.7783 | 2127222078 | 1655650211 |
| fp8_linear | 13 | 0.9992 | 7368729473 | 7361051726 |
| gemv | 1 | 0.8398 | 218449056 | 183451040 |
| linear | 1 | 1.0368 | 1241624471 | 1287266695 |
| rmsnorm_residual | 1 | 0.5054 | 20752010 | 10488022 |
| rope | 1 | 0.5695 | 36185002 | 20606154 |
| swiglu | 1 | 0.4889 | 47210443 | 23082387 |

| point | old cycles | new cycles | new/old | samples old/new | err/tol (new) |
|---|--:|--:|--:|--:|--:|
| `model.fp8_decoder_decode/xl/fp8-b1-p2048-layers28` | 2485589764 | 1726718215 | 0.6947 | 1/1 | 0.202 |
| `model.fp8_decoder_prefill/xl/fp8-b1-s2048-layers4` | 63572263224 | 34531184415 | 0.5432 | 1/1 | 0.048 |
| `model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28` | 60103313196 | 47092602832 | 0.7835 | 1/1 | 0.198 |
| `model.fp8_layer_decode/xl/fp8-b1-p2048-layers1` | 78866886 | 51604507 | 0.6543 | 3/3 | 0.158 |
| `model.fp8_layer_prefill/xl/fp8-b1-s512-layers1` | 2127222078 | 1655650211 | 0.7783 | 1/1 | 0.141 |
| `op.argmax/xl/fp16-r16-v151936` | 4614775 | 2733845 | 0.5924 | 3/3 | 0.000 |
| `op.attention_decode/xl/fp16-b1-h16-kv8-l2048-d128` | 28609797 | 19425345 | 0.6790 | 3/3 | 0.002 |
| `op.attention_decode/xl/fp16-b1-h16-kv8-l4096-d128` | 57935733 | 38678944 | 0.6676 | 3/3 | 0.001 |
| `op.attention_decode/xl/fp16-b1-h16-kv8-l8192-d128` | 115684393 | 76531974 | 0.6616 | 1/3 | 0.001 |
| `op.attention_prefill/xl/fp16-b1-h16-kv8-s1024-d128` | 2451582300 | 644201889 | 0.2628 | 1/1 | 0.014 |
| `op.attention_prefill/xl/fp16-b1-h16-kv8-s2048-d128` | 9758617442 | 2545596033 | 0.2609 | 1/1 | 0.012 |
| `op.embedding/xl/fp16-v151936-d2048-t2048` | 74462105 | 74460826 | 1.0000 | 3/3 | 0.000 |
| `op.fp8_act_quant/xl/fp8-m2048-k2048` | 6552352 | 6551944 | 0.9999 | 3/3 | 0.893 |
| `op.fp8_act_quant/xl/fp8-m2048-k6144` | 19825345 | 19825411 | 1.0000 | 3/3 | 0.893 |
| `op.fp8_linear/xl/fp8-m1-k2048-n12288` | 12441193 | 12417959 | 0.9981 | 3/3 | 0.008 |
| `op.fp8_linear/xl/fp8-m1-k2048-n151936` | 152193288 | 153090135 | 1.0059 | 1/1 | 0.116 |
| `op.fp8_linear/xl/fp8-m1-k2048-n4096` | 4233933 | 4228249 | 0.9987 | 3/3 | 0.000 |
| `op.fp8_linear/xl/fp8-m1-k6144-n2048` | 7094096 | 7035586 | 0.9918 | 3/3 | 0.006 |
| `op.fp8_linear/xl/fp8-m16-k2048-n12288` | 21509187 | 21512812 | 1.0002 | 3/3 | 0.052 |
| `op.fp8_linear/xl/fp8-m16-k2048-n151936` | 265576929 | 265585076 | 1.0000 | 1/1 | 0.093 |
| `op.fp8_linear/xl/fp8-m16-k6144-n2048` | 12322561 | 12317074 | 0.9996 | 3/3 | 0.097 |
| `op.fp8_linear/xl/fp8-m2048-k2048-n12288` | 2883696822 | 2878935510 | 0.9983 | 1/1 | 0.086 |
| `op.fp8_linear/xl/fp8-m2048-k2048-n2048` | 480684762 | 481956411 | 1.0026 | 1/1 | 0.098 |
| `op.fp8_linear/xl/fp8-m2048-k2048-n4096` | 960073979 | 959524103 | 0.9994 | 1/1 | 0.193 |
| `op.fp8_linear/xl/fp8-m2048-k6144-n2048` | 1474501995 | 1471238004 | 0.9978 | 1/1 | 0.095 |
| `op.fp8_linear/xl/fp8-m512-k2048-n12288` | 722379652 | 722291085 | 0.9999 | 1/1 | 0.091 |
| `op.fp8_linear/xl/fp8-m512-k6144-n2048` | 372021076 | 370919722 | 0.9970 | 1/1 | 0.095 |
| `op.gemv/xl/fp16-k2048-n151936` | 218449056 | 183451040 | 0.8398 | 1/1 | 0.020 |
| `op.linear/xl/fp16-m2048-k2048-n6144-bias0` | 1241624471 | 1287266695 | 1.0368 | 1/1 | 0.017 |
| `op.rmsnorm_residual/xl/fp16-r2048-d2048` | 20752010 | 10488022 | 0.5054 | 3/3 | 0.029 |
| `op.rope/xl/fp16-b1-h16-s2048-d128` | 36185002 | 20606154 | 0.5695 | 3/3 | 0.019 |
| `op.swiglu/xl/fp16-r2048-i6144` | 47210443 | 23082387 | 0.4889 | 3/3 | 0.014 |
