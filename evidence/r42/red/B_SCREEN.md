# Part B (with part A): screen of the R42-red head against unchanged main

Domain: **single-core RTL simulation** (diagnostic screen, not a board result), profile
`core-vrf-sync-v5-x8-g2c4-dp`, default knobs, deterministic serial-grid device cycles. Columns: point, base
cycles, head cycles, base/head. Tree: commit 65b42be4 (GB10 `~/r42/red_out/trees/D`), PYTHONPATH =
python:compiler/python:tools/asm of the tree. Command per run: `~/r42/red_out/run.sh TREE PROFILE JOBS OUT ...`
= `python -m archbench.run --platform tnca --profile core-vrf-sync-v5-x8-g2c4-dp [--tier tiny | --point ...]`.

## Tiny tier (27 points; base = `records/base_dp_tiny_thor.jsonl`, head = `records/D1_tiny.jsonl`)

`python3 r42_cmp.py records/base_dp_tiny_thor.jsonl records/D1_tiny.jsonl`
```
model.gpt2_block/tiny/fp16-d64-h4-b1-s16                           232013     207494  1.118
model.llama_decode/tiny/fp16-b1-p16                                515903     454059  1.136
model.llama_prefill/tiny/fp16-b1-s16                               701695     673260  1.042
model.mlp/tiny/fp16-t16-d64                                         84918      81403  1.043
model.resnet_block/tiny/fp32-n1-c8-hw8                             328602     328828  0.999
op.add/tiny/fp16-n1000                                               2354       2370  0.993
op.add/tiny/fp32-n256                                                 775        825  0.939
op.argmax/tiny/fp16-r2-v256                                          2934       2034  1.442
op.attention_decode/tiny/fp16-b1-h2-l16-d16                          9787       9967  0.982
op.attention_prefill/tiny/fp16-b1-h2-s16-d16                        40707      39490  1.031
op.bmm/tiny/fp16-b2-m16-k16-n16                                      1924       1856  1.037
op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8                            145168     145027  1.001
op.cumsum/tiny/fp32-r4-c64                                           2505       2218  1.129
op.embedding/tiny/fp16-v256-d64-t16                                 46381      46381  1.000
op.gelu_tanh/tiny/fp16-n256                                          2228       1972  1.130
op.gemv/tiny/fp16-k64-n64                                            8684       8603  1.009
op.layernorm/tiny/fp16-r4-d64                                        3384       2043  1.656
op.linear/tiny/fp16-m16-k64-n64                                      6265       6481  0.967
op.matmul/tiny/fp16-m16-k32-n16                                      1214       1211  1.002
op.matmul/tiny/fp32-m16-k32-n16                                      4921       4918  1.001
op.rmsnorm/tiny/fp16-r4-d64                                          2719       1406  1.934
op.rope/tiny/fp16-b1-h2-s16-d16                                      4313       4265  1.011
op.silu/tiny/fp16-n256                                               1595       1462  1.091
op.softmax/tiny/fp16-r4-c64                                          3875       2465  1.572
op.softmax/tiny/fp32-r3-c37                                          1809       1340  1.350
op.swiglu/tiny/fp16-n256                                             1699       1660  1.023
op.transpose/tiny/fp32-r32-c32                                       2310       2310  1.000
points 27/27  S_geo 1.1146  sum-ratio 1.0616  slower>2%: 2
```
All 27 pass numerically. Slower by more than 2 %:
- `op.add/tiny/fp32-n256` 0.939: part A (see A_TILED_DEFAULT.md): per-tile lane indices and masks of the tiled
  program on a 256-element input. (`op.add/tiny/fp16-n1000` and `op.rope/tiny`, 0.936/0.937 with part A alone,
  are 0.993/1.011 at the head.)
- `op.linear/tiny/fp16-m16-k64-n64` 0.967 (6,265 -> 6,481): the addmm template kernel has 392 instructions
  instead of 481 (the loop pass rewrote its scalar loop bookkeeping; no transfer or matrix command changed), yet
  runs 216 cycles longer. The cause inside the transfer/MMA timing was not identified. The other template points
  moved the other way or not at all (bmm 1.037, gemv 1.009, matmul 1.002/1.001, conv2d 1.001, mlp 1.043).

## Small-tier points of the changed families (base = `records/base_small_family.jsonl`, head = `records/D1_small_family.jsonl`)

`python3 r42_cmp.py records/base_small_family.jsonl records/D1_small_family.jsonl`
```
op.add/small/fp16-n16384                                            33897      28956  1.171
op.argmax/small/fp16-r8-v2048                                       63301      37202  1.702
op.cumsum/small/fp32-r64-c256                                      150012     128229  1.170
op.gelu_tanh/small/fp16-n16384                                     129911      85608  1.518
op.layernorm/small/fp16-r64-d256                                   178212      80475  2.215
op.rmsnorm/small/fp16-r64-d256                                     138891      48284  2.877
op.rope/small/fp16-b1-h4-s64-d32                                    59910      47839  1.252
op.silu/small/fp16-n16384                                           90756      88952  1.020
op.softmax/small/fp16-r64-c512                                     449972     170919  2.633
op.swiglu/small/fp16-n16384                                         81953      95839  0.855
points 10/10  S_geo 1.5176  sum-ratio 1.6950  slower>2%: 1
```
All pass numerically. `op.swiglu/small` is slower (0.855); the tiled lowering of part A is the only change that
reaches this kernel class; the A-only control (`records/A_smallpoi.jsonl`) is recorded when it finishes.

## Gates
- `make compiler` (GB10, tree D): PASS, `compiler-unit` 182 tests (67 skipped), ISA/target/datapath checks PASS;
  `compiler-ir` SKIP because tnca-opt is not built on GB10 (`records/mk_compiler_D.log`).
- No RTL changed. Not run: the 113-point compile-only default-change audit, the small tier beyond the points
  above, the three-core stand-in.
