# b41-run3-base: the 113 original points on the R41 run-3 image with unchanged `main` software (board)

Domain: board, U280, image `791b8af2…` (UUID `ea1fa1af-…`), measured 100 MHz, `device-cycles-serial-grid`.
Registration: [`spec/r42/board/b41-run3-base/`](../../../../spec/r42/board/b41-run3-base/) (commit `91cad39e`, pushed before the run).
Software: `main` `4b61abd8`, tree `files:9490a5f5…`, staged as `~/r42/src_base` on the FPGA host. Recipe: [`../RECIPE.md`](../RECIPE.md).

- 2026-11-20 08:12 UTC runner started (`~/r42/q/b41-run3-base`, PID 2171995). The card held image `1c27e9d1-…` with no
  context; after 901 s of continuous idle the runner loaded the run-3 image with `program.sh --check` / `--go`
  (08:27:08-08:27:18 UTC, `program.log` in the archive; `program.log.previous_load` is the R41 log that `program.sh` overwrites).
- Identity (after a further 15 idle minutes): clock verification pass, 3/3 points pass.
- Suite (after a further 15 idle minutes, 08:59-10:27 UTC): 113/113 configurations, 205 runs (113 qualification + 92
  three-sample formal), no failed run, 5,278 s of process wall time.
- Records: [`records.tgz`](records.tgz), sha256 `5232d3ed…` (events, logs, reports, `batch.json`, `state.json`,
  `formal.jsonl`, every raw run record; bundles and caches stay on the host).

Comparison with the R41 run-3 records (`tools/r42/board_compare.py`, [`comparison.json`](comparison.json)):
geometric mean new/old 0.9995 over 113 points (median 1.0000, min 0.9695, max 1.0190); by tier hbm 1.0035, large 1.0003,
medium 0.9994, small 0.9992, tiny 0.9975. Two points differ by more than 2 %: `model.llama_prefill/large/fp16-b1-s512`
0.9695 (one sample on each side) and `op.silu/tiny/fp16-n256` 0.9800 (1,149 -> 1,126 cycles). No systematic difference
between the R41 software of 2026-10-30 and `main`. The recorded artifact hashes differ for every point between the two
trees; whether the generated code itself differs was not examined.
