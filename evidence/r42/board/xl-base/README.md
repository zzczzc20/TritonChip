# xl-base: the 32 xl points on the R41 run-3 image, baseline software (board)

Domain: board, U280, image `791b8af2...` (UUID `ea1fa1af-...`), measured 100 MHz, `device-cycles-serial-grid`, run-3 knobs
(`TNCA_ACTIVE_CORES=3`, `TNCA_TORCH_DEVICE=native`, `TNCA_TRANSFER_POLICY=resident`), default compiler settings.
Registration: [`spec/r42/board/xl-base/`](../../../../spec/r42/board/xl-base/) (commit `fd524362`, pushed before the run). Tool: `tools/r42/board_ab.py register --suite xl`,
`tools/r42/xl_board_batch.py` (registered per-point timeouts and the 300 s sampling cap; the cap did not trigger).
Software: branch `R42-xl-base` `5e5236b4` = `main` `4b61abd8` plus only the `archbench/` additions of `R42-xl` (xl tier, FP8 point types, vendored Hub kernels,
launch-count fix), its `tools/r42/xl_*` helpers and `doc/r42_plan.md`; `git diff --stat 4b61abd8 R42-xl-base -- compiler python rtl` is empty.
Tree `files:64bf16ee...`, staged as `~/r42/src_xlbase` on the FPGA host `acf3019`; queue directory `~/r42/q/xl-base`.

Timeline (UTC, 2026-11-23, from `events.jsonl`): runner launched 16:44, stopped before any card access (the card held another user's image, instruction not to load);
resumed 16:52, stopped 19:58 after 3 h without 15 idle minutes (2103 idle probes, longest idle stretch 115 s); resumed 19:58;
15 consecutive idle minutes met 20:24:02 (previous image `d6bd7770-...`, no context), `program.sh --check` then `--go` once, run-3 image verified at 100 MHz 20:24:14;
identity manifest 20:39-20:41 (clock verification pass, 3/3 points pass); xl suite 20:55:47 - 22:23:08 (queue-end rc 0).

Result: 32/32 configurations pass (numerics inside the frozen tolerance, no host fallback), 47 runs (32 qualification + 15 three-sample formal), no failed run, 5,240 s of process wall time.
Records: [`records.tgz`](records.tgz) sha256 `2764d40a...`, per-run process logs [`logs.tgz`](logs.tgz), summary [`summary.json`](summary.json) (`tools/r42/board_compare_xl.py --new records.tgz`).

| family | points | device seconds (formal records) | geometric mean eta (bound / measured) |
|---|--:|--:|--:|
| argmax | 1 | 0.05 | 0.132 |
| attention_decode | 3 | 2.02 | 0.036 |
| attention_prefill | 2 | 122.10 | 0.028 |
| embedding | 1 | 0.74 | 0.028 |
| fp8_act_quant | 2 | 0.26 | 0.242 |
| fp8_decoder_decode | 1 | 24.86 | 0.114 |
| fp8_decoder_prefill | 2 | 1236.76 | 0.117 |
| fp8_layer_decode | 1 | 0.79 | 0.093 |
| fp8_layer_prefill | 1 | 21.27 | 0.156 |
| fp8_linear | 13 | 73.69 | 0.230 |
| gemv | 1 | 2.18 | 0.356 |
| linear | 1 | 12.42 | 0.649 |
| rmsnorm_residual | 1 | 0.21 | 0.202 |
| rope | 1 | 0.36 | 0.062 |
| swiglu | 1 | 0.47 | 0.200 |

Largest heap peak 2272010496 bytes against a heap capacity of 8,589,930,496; largest error / tolerance 0.893 (`op.fp8_act_quant/xl/fp8-m2048-k2048` and `-k6144`: 0.893).
