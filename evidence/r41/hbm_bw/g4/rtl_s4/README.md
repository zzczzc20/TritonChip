# RTL link probes on the candidate image (s4)

**Domain: RTL simulation** of `tnca_multicore_hbm_xo` in the board bench, with the testbench memory model. These are not board measurements. Points are 64 KiB registered points. Tool: `tools/r41/link_budget_rtl.py --image … --profile core-vrf-sync-v5-x8-g2 --point …`. Mode 1/2 records were made with the NON-DEFAULT `--set TNCA_TILED_POINTWISE=N`.

| Directory | Image | Compiler | Points |
|---|---|---|---|
| `old_compiler/` | `g4/images/image_s4.tcl` | software tree `5c30cd0a…` (before the grouped lowering): mode 0 (default), mode 1 (fp32 points), mode 2 | 20 / 4 / 20 |
| `grouped/` | `image_s4.tcl` | software tree `21fbf302…` (`2a134be`, grouped lowering), mode 2 | 20 |
| `controls/` | `image_g2.tcl`, `image_s1.tcl` | `21fbf302…`, mode 0 | the five points slower than run 1 under the default |

Every point passed its numerical check (`w*.log`). `rtl_table.md` is the output of:

```sh
python3 tools/r41/g4_rtl_report.py --arm "g2 m0=evidence/r41/hbm_bw/g4/rtl_s4/controls/g2.mode0*.jsonl" \
  --arm "s1 m0=…/controls/s1.mode0*.jsonl" --arm "s4 m0=…/old_compiler/s4.mode0*.jsonl" \
  --arm "s4 m1=…/old_compiler/s4.mode1*.jsonl" --arm "s4 m2 earlier compiler=…/old_compiler/s4.mode2*.jsonl" \
  --arm "s4 m2=…/grouped/*.jsonl"
```
The board bench at the time of these runs had the shared-BO mirror defect described in `../geometry/README.md`. It does not affect these records, for two reasons. The defect only matters when one line is written through one outlet and read through another, which happens only with the single-outlet memory control, and these runs use all three outlets. In addition, each record passed its numerical check against the host reference.
