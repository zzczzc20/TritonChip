# Part A: `TNCA_TILED_POINTWISE` default 0 -> 2

Domain: **single-core RTL simulation** (a diagnostic screen, not a board result), profile
`core-vrf-sync-v5-x8-g2c4-dp`, ArchBench tiny tier, default knobs, deterministic device cycles
(serial-grid, one sample). Columns: point, base cycles, candidate cycles, base/candidate.

- base: unchanged main (`~/r42/base`), measured by the lead on thor with the corrected import path
  (`records/base_dp_tiny_thor.jsonl`); six of its points were re-measured here on GB10 and are identical
  (`records/base_tinyred.jsonl`).
- candidate: this commit (`~/r42/red` on GB10), `~/r42/red_out/run.sh red core-vrf-sync-v5-x8-g2c4-dp 2 A_tiny --tier tiny`
  (`python -m archbench.run --platform tnca --profile ... --tier tiny`, PYTHONPATH = python:compiler/python:tools/asm of the tree),
  `records/A_tiny.jsonl`. All 27 points pass numerically.
- table: `python3 r42_cmp.py records/base_dp_tiny_thor.jsonl records/A_tiny.jsonl`

```
model.gpt2_block/tiny/fp16-d64-h4-b1-s16                           232013     230839  1.005
model.llama_decode/tiny/fp16-b1-p16                                515903     516071  1.000
model.llama_prefill/tiny/fp16-b1-s16                               701695     700205  1.002
model.mlp/tiny/fp16-t16-d64                                         84918      84918  1.000
model.resnet_block/tiny/fp32-n1-c8-hw8                             328602     329245  0.998
op.add/tiny/fp16-n1000                                               2354       2516  0.936
op.add/tiny/fp32-n256                                                 775        825  0.939
op.argmax/tiny/fp16-r2-v256                                          2934       2934  1.000
op.attention_decode/tiny/fp16-b1-h2-l16-d16                          9787       9787  1.000
op.attention_prefill/tiny/fp16-b1-h2-s16-d16                        40707      40429  1.007
op.bmm/tiny/fp16-b2-m16-k16-n16                                      1924       1924  1.000
op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8                            145168     145168  1.000
op.cumsum/tiny/fp32-r4-c64                                           2505       2505  1.000
op.embedding/tiny/fp16-v256-d64-t16                                 46381      46381  1.000
op.gelu_tanh/tiny/fp16-n256                                          2228       1972  1.130
op.gemv/tiny/fp16-k64-n64                                            8684       8684  1.000
op.layernorm/tiny/fp16-r4-d64                                        3384       3384  1.000
op.linear/tiny/fp16-m16-k64-n64                                      6265       6265  1.000
op.matmul/tiny/fp16-m16-k32-n16                                      1214       1214  1.000
op.matmul/tiny/fp32-m16-k32-n16                                      4921       4921  1.000
op.rmsnorm/tiny/fp16-r4-d64                                          2719       2719  1.000
op.rope/tiny/fp16-b1-h2-s16-d16                                      4313       4604  0.937
op.silu/tiny/fp16-n256                                               1595       1462  1.091
op.softmax/tiny/fp16-r4-c64                                          3875       3875  1.000
op.softmax/tiny/fp32-r3-c37                                          1809       1809  1.000
op.swiglu/tiny/fp16-n256                                             1699       1660  1.023
op.transpose/tiny/fp32-r32-c32                                       2310       2310  1.000
points 27/27  S_geo 1.0018  sum-ratio 1.0010  slower>2%: 3
```

Points slower by more than 2 % and their cause (from the dumped assembly, `tools`: TNCA_KERNEL_DUMP):
- `op.add/tiny/fp16-n1000` 0.936, `op.add/tiny/fp32-n256` 0.939, `op.rope/tiny` 0.937. The tiled program is a
  loop over 64-lane tiles that rebuilds, per tile, the lane index vectors and the `index < xnumel` masks of
  every register chunk (8 `v.cmp`, 8 `v.add`, 2 `v.splat` per two-tile iteration in the fp16 add) and materialises
  `other=0` before every masked load; on an input of a few hundred elements there are too few tiles to amortise
  that and the larger program prologue. The mask cannot be dropped at compile time: it depends on the grid.
- The gain of mode 2 measured in R41 (1.489x RTL, 1.578x board) was on 64 KiB and larger batches on other images.
  On this profile the small tier (16,384 elements; base against the R42-red head, which contains this change;
  `records/base_smallpoi.jsonl`, `records/D1_smallpoi.jsonl`) gives gelu_tanh 1.517, add 1.171, rope 1.252,
  silu 1.020, swiglu 0.855. The swiglu loss is not explained yet (see ANALYSIS.md).

`make compiler` on GB10 from the tree: PASS, 173 tests, 67 skipped; `compiler-ir` SKIP (tnca-opt is not built on GB10).
The compile-only default-change audit (113 points) was not run.
