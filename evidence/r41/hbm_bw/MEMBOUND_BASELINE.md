# Board baseline of the registered memory-bound suite (run-1 image, 100 MHz)

**Domain: board.** Image `30e79769…` (UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`), 100 MHz, `fpga-u280-multicore`, standard path with default knobs. [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md) is binding.

This is the baseline that [`LINK_BUDGET.md`](LINK_BUDGET.md) §1 registers for the frozen suite `archbench/suites/r41_membound.json` (513 points). It is the reference for the 32 GB/s target of [`BANDWIDTH_PLAN.md`](BANDWIDTH_PLAN.md).

## Run
- **Who and where:** run by the lead on acf3019, with `evidence/r41/hbm_bw/launches/membound_baseline.sh`.
- **Source:** `~/r41_membound_e57b64c` (commit e57b64c).
- **Output:** `~/r41_membound_e57b64c_out`.
- **Time:** 2026-10-28, 12:52Z to 21:15Z. All six slices exited 0, the log ends with `MEMBOUND-BASELINE-DONE`, and the batches account for 6.87 board hours.
- **Archive in the repository:** `membound_baseline/membound_baseline_e57b64c.tgz`, sha256 `a90221ea93b64d3dce04bab3e5217ce5ac0975a2a27e8ea4d3021e7205e988d3` (`membound_baseline/SHA256SUMS`). It holds every slice's `batch.json`, `state.json`, `formal.jsonl`, run records, logs, experiment reports, and the kernel assembly of each provenance bundle. Excluded: compiler caches, Triton caches, binaries, intermediate IR.
- **Full copy on the host:** `acf3019:~/r41_membound_e57b64c_out_full.tgz`, sha256 `6ac88a980de4125d6c4f42061ce2585bfb62773355cdbc285889af00393cc2ef`. It excludes only compiler caches.

## Tables
Every table is generated from the archive by `python3 tools/r41/membound_baseline_table.py --write`:
- `membound_baseline/table.json`: every point and every summary;
- `membound_baseline/tables.md`: all tables, including all 513 points.

For each point:
- device time is the median of the formal serial-grid samples divided by the record's clock (100 MHz);
- full-link GB/s is ArchBench demand reads plus writes divided by that time;
- eta comes from the record (ArchBench bound over device cycles).

## Result
- **Status:** 513 of 513 points pass numerically on the board. No failed run, no missing formal record and no definition mismatch: every batch is `complete`, with `failed_configurations` and `missing_formal` empty.
- **Best:** 0.31 GB/s (fp32 contiguous `sum`, `max`, `mean`, large).
- **Median:** 0.138 GB/s at the large size (0.130 at small).
- **Target:** 32 GB/s.
- **Agreement with the earlier run:** fp16 contiguous large `add` gives 0.158 GB/s, against 0.161 GB/s for the run-1 `op.add/hbm` point of the original suite.

Large size (64 MiB per full-size tensor), GB/s per family:

| Family | Pass | Best | Median | Worst | Best point | Worst point |
|---|---:|---:|---:|---:|---|---|
| add | 9/9 | 0.2483 | 0.1583 | 0.115 | `fp32-large-contig` | `fp16-large-broadcast` |
| sub | 9/9 | 0.2484 | 0.1583 | 0.1149 | `fp32-large-contig` | `fp16-large-broadcast` |
| mul | 9/9 | 0.2484 | 0.1583 | 0.1149 | `fp32-large-contig` | `bf16-large-broadcast` |
| div | 9/9 | 0.242 | 0.1489 | 0.1078 | `fp32-large-contig` | `fp16-large-broadcast` |
| copy | 9/9 | 0.2496 | 0.1396 | 0.08518 | `fp32-large-contig` | `bf16-large-broadcast` |
| clone | 9/9 | 0.2496 | 0.1397 | 0.08513 | `fp32-large-contig` | `bf16-large-broadcast` |
| relu | 9/9 | 0.2497 | 0.1397 | 0.08378 | `fp32-large-contig` | `bf16-large-broadcast` |
| cast | 18/18 | 0.2097 | 0.1475 | 0.08294 | `fp32-to-fp16-large-contig` | `fp32-to-fp16-large-broadcast` |
| exp | 9/9 | 0.2357 | 0.1222 | 0.06814 | `fp32-large-contig` | `fp16-large-broadcast` |
| sigmoid | 9/9 | 0.1982 | 0.09856 | 0.05299 | `fp32-large-contig` | `bf16-large-broadcast` |
| silu | 9/9 | 0.1987 | 0.0987 | 0.05268 | `fp32-large-contig` | `bf16-large-broadcast` |
| gelu | 9/9 | 0.1004 | 0.05022 | 0.02485 | `fp32-large-contig` | `fp16-large-broadcast` |
| sum | 9/9 | 0.3125 | 0.1403 | 0.0003206 | `fp32-large-contig` | `bf16-large-broadcast` |
| mean | 9/9 | 0.3114 | 0.1398 | 0.000315 | `fp32-large-contig` | `bf16-large-broadcast` |
| max | 9/9 | 0.3124 | 0.1403 | 0.0003186 | `fp32-large-contig` | `fp16-large-broadcast` |
| rmsnorm | 9/9 | 0.1625 | 0.09945 | 0.0654 | `fp32-large-contig` | `fp16-large-broadcast` |
| layernorm | 9/9 | 0.1222 | 0.07242 | 0.05179 | `fp32-large-contig` | `bf16-large-broadcast` |
| softmax | 9/9 | 0.1195 | 0.06449 | 0.0325 | `fp32-large-contig` | `bf16-large-broadcast` |

By layout, at the large size, median GB/s:

| Layout | Median GB/s |
|---|---:|
| contiguous | 0.158 |
| strided | 0.138 |
| broadcast | 0.084 |

What the numbers show:
- **fp32 is fastest in every family.** One 16-lane program moves twice the bytes of an fp16 program at a similar cost per program, which matches the per-program cost found in LINK_BUDGET.md §2–3.
- **The cost per program does not depend on size.** Medium and large are almost identical, and small differs only slightly.
- **The broadcast reductions are near zero by definition, not by a fault.** Their registered demand is one 1024-element row read plus R outputs written, while the kernels run over every row. Full-link GB/s therefore counts only those few bytes.

## Use
- This baseline is the registered reference for the bandwidth target, measured with the same harness and the same timing boundary as any later image (discipline §2.3.3).
- The non-regression baseline of the 113 original points stays the `s2-original-suite` run-1 record.
- In LINK_BUDGET.md, the "measured end to end" row of the stage table is the board anchor; this file extends it to every registered point.
