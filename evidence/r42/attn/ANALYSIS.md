# R42 `attn`: where the cycles of attention and bmm go

Domain of every number here: **single-core RTL simulation** (profile `core-vrf-sync-v5-x8-g2c4-dp`, the profile
of the cores of the R41 run-3 image), device cycles of the serial grid, default knobs. It is a diagnostic screen,
not a board result. Baseline = unchanged `main` (`~/r42/base` on GB10); records in this directory.
Kernel dumps (`TNCA_KERNEL_DUMP=1`, one directory per point) were read for the tiny and small points of
`attention_prefill`, `attention_decode` and `bmm`.

## 1. What Inductor generates

`F.scaled_dot_product_attention` on an FP16 CPU/tnca tensor reaches Inductor as
`aten._scaled_dot_product_flash_attention_for_cpu`; `tnca.inductor._decompose_cpu_attention` adds upstream's own
decomposition (`_scaled_dot_product_attention_math`). Upstream's math path **converts q, k, v to float32 first**
and converts the result back (`make_fx` of the op, torch 2.14: three `_to_copy(dtype=float32)`, `mul.Scalar`,
`bmm`, `_safe_softmax`, `bmm`, `_to_copy(float16)`). So every matrix product of an attention is a **binary32**
`tl.dot`, and ISA sec 9C has no binary32 matrix command: the products run on vector FMAs, not on the MMA.
That, not the softmax, is the structural reason attention is far from `linear`/`matmul`.

Prefill (causal), seven launches:

| kernel | what | grid (tiny) |
|---|---|---|
| `poi_0` | `q.float() * scale` | 32 |
| `poi_1` | `k.float().transpose(-2,-1) * scale` (2-D transposed gather) | 1x2 |
| `tem_2` | `bmm` (f32) scores | 1x2 |
| `red_3` | causal mask + `_safe_softmax` (max / any / exp / sum / div), one row a program | 32 |
| `poi_4` | `v.float()` | 32 |
| `tem_2` (small: `tem_5`) | `bmm` (f32) probabilities x values | 1x2 |
| `poi_5` | `.half()` + the permute/contiguous of upstream's decomposition | 32 |

Decode (one query row): upstream's `tuned_bmm` lowers a `bmm` with M = 1 on a `cpu`-type device as
`mul` + `sum`, so there is **no GEMM kernel at all**: three generic looped reductions (`red_0` q.k per score,
`red_1` softmax statistics, `red_2` probabilities x V with a stride-d gather per output element), one program
per output scalar.

## 2. Baseline per-kernel cycles (tiny, `base3.jsonl`)

`op.attention_prefill/tiny` 40,707 cycles:

| kernel | cycles | share |
|---|---:|---:|
| `red_3` softmax | 20,091 | 49 % |
| `tem_2` f32 bmm, twice | 2 x 4,824 | 24 % |
| `poi_1` transposed scale | 5,610 | 14 % |
| `poi_5` | 3,056 | 8 % |
| `poi_0`, `poi_4` | 1,280 + 1,022 | 6 % |

`op.attention_decode/tiny` 9,787: `red_0` 2,988, `red_1` 1,064, `red_2` 5,735.
`op.bmm/tiny` 1,924: one launch, two programs of one MMA each.

## 3. Findings (dominant kernels, from the `.ttir` / `.s`)

**F1. A 16-lane int64 mask costs about 400 instructions.** `red_3` is 577 instructions a program for one
16-element row (628 cycles a row). Inductor writes the causal mask as `(col - row).to(int64) <= 0` and the
"any finite score" flag as `(~(x == -inf)).to(int64) != 0`. `tnca_cc.wide` holds a 64-bit block as low/high
word pairs and does everything a lane at a time in scalar registers: the widening is extract + 2 shifts +
2 inserts a lane, the compare 2 extracts + shift + or + scalar compare + insert a lane. Three such pairs in
the program: ~400 of the 577 instructions. The arithmetic that remains (load, add, max, exp, sum, reciprocal,
store) is ~140 instructions.

**F2. A GEMM template that carries its pointers round the K loop falls onto index tiles and gathers.** The
`bmm` template still advances `A += BLOCK_K * stride` in the loop (the `mm` template recomputes the pointer
from `k_idx`). `_op_scf_for` carried a 2-D pointer as a 16x16 int32 index tile in scratch memory: every
iteration re-read 16 index rows, gathered 16 rows (`v.ld.gather.h`), stored them to scratch, repacked them to
FP16, built a new TensorMap (14 stores, a fence, `td.load`) and only then transferred the tile into the window
the MMA reads -- for A and for B -- and rewrote both index tiles (32 loads, 32 adds, 32 stores). `op.bmm/small`:
1,282 instructions, 32 gathers and 10 device fences per program, against 410 instructions for the same loop
in the `mm` template. With K = 16 (tiny) there is no loop, which is why only bmm/tiny looks like matmul
(eta 0.25 vs 0.065 for small/medium/large on the board). The f32 attention bmm takes the same path whenever
K > 16 (every tier above tiny, and every model).

**F3. The f32 dot is a 16-deep dependent FMA chain per row, issued in order.** `_dot_f32` emitted, row after
row, 16 x (`v.extract`, `v.splat`, `vf.fma`) with each FMA reading the previous one: the core issues in order,
so each FMA waited for its predecessor (FP latency 5). 2,412 cycles per 16x16x16 tile = 0.59 cycles/MAC, and
three instructions per 16 MACs even without the stall. Inside a K loop the dot also went to an accumulator of
its own (16 `a.vwrite`), was read back, added to the carried accumulator (32 `a.vread`, 16 `vf.add`) and copied
back by the yield (16 `a.vwrite`).

**F4. One program per row / per output scalar is latency-bound.** After F1 the softmax row is a serial chain
load -> add -> max -> sub -> exp2 (SFU, 28) -> sum -> rcp (SFU, 28) -> Newton -> mul -> store: about 245
cycles a row however short the row is; programs of one context run one after the other, so nothing overlaps.
The decode reductions are the same shape: 93 cycles for a 16-MAC dot (`red_0`), 179 for a 16-element strided
column (`red_2`), with unit-dimension index arithmetic (`tensor<1xi32>` division and remainder) done in vector
registers. These are the generic reduction lowering (owner: worker `red`); TNCA_REDUCTION_ROWS > 1 is the
existing mechanism that would interleave rows and turn `red_2`'s column gathers into transposed tile loads.

**F5. Pointwise parts.** `poi_1` (transposed f16 -> f32 scale) is 862 instructions per 16x16 tile, 11
cycles/element: the 2-D index tile is built in scratch and gathered. `poi_5` gathers through a
remainder/divide index. Both are element-wise lowering (TNCA_TILED_POINTWISE, another worker).

**F6. bmm/tiny (no loop):** 962 cycles a program for one MMA: ~45 % is the epilogue (16 x `a.vread`,
`vf.add 0`, convert, three recomputed masks, `v.st.h` through the LSU), the rest the two transfers, the
software divide prologue and the MMA. Not changed here.
