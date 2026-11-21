# Commands behind the records in this directory

Host: GB10, trees `~/r42/base` (unchanged main) and `~/r42/attn` (this branch, mirrored with `r42_sync`).
Environment of every run: `source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm`,
`rm -rf python/tnca/data`, nice 19, profile `core-vrf-sync-v5-x8-g2c4-dp`, default knobs.
Domain: single-core RTL simulation (diagnostic screen).

- `attn_pts.sh TREE OUT JOBS POINT...`: one `python -m archbench.run --platform tnca --profile P --point ID` per
  point, records concatenated to `~/r42/attn_out/OUT.jsonl`.
- `attn_dump.sh TREE TAG POINT...`: `--compile-only` with `TNCA_KERNEL_DUMP=1`, one dump directory per point.
- `attn_show.py F.jsonl`: point status, device cycles and `time.per_launch`.
- `tiny_pts.txt`: the 27 tiny-tier points.
- comparison: `python3 ~/r42/r42_cmp.py BASE.jsonl CAND.jsonl`.
