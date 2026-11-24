# R42-copy: why embedding, transpose and cumsum did not move, and the general lowering changes

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Branch `R42-copy` (from `origin/R42`
> `02b7578e`), 2026-11-23/24. Every number names its domain: **board** = `ab-int2` records only (quoted, nothing new is measured
> on the board here); **rtl** = single-core RTL simulation, profile `core-vrf-sync-v5-x8-g2c4-dp`, a screen; **tlm** = the
> SystemC/TLM model of the three-core image (`tools/r42/kfuse_speed_tlm_run.py`, the calibrated `cal.cfg` of `R42-kfuse-speed`),
> a model prediction; **census** = instructions in the assembly (code size, both sides of a branch counted). Raw records:
> `records/`; the tables below are `python3 evidence/r42/copy/tables.py` over them; commands: `commands.txt`.

## 1. Why the compiled stream was not faster (diagnosis)

Board numbers (`ab-int2`): embedding 1.00x (5 points), transpose 1.00x (5), cumsum 1.17x (3).

**embedding** (`F.embedding`, Inductor kernel `weight[ids[x // d] * d + x % d]`, 16 elements a program).
The program was 704 instructions (`op.embedding/tiny`, census of the R42 head): the token id and every index derived from it is a
64-bit block, which `wide.py` holds as (low, high) vector pairs and computes **lane by lane in scalar registers** -- an extract,
an or, a shift and an insert per lane for every operation (the two bound compares of the assert, the multiply by `d`, the
narrowing of the address), two gathers for the id (`x // d` was not known to be the same in all lanes, because a program id was not known to be
non-negative, so it was divided as a vector), and one gather for the data (one memory word a lane): about 44 instructions a lane.
On the board that is 285 cycles for each 16 elements (299.5 M cycles for `hbm`: 1,048,576 programs); in the single-core RTL 95 % of the
cycles are issue slots (`perf_issued` 44,088 of 46,384, `op.embedding/tiny`). The widest config the planner offers (`XBLOCK` up to 1,024)
failed on `tt.assert` ("wide register block operation tt.assert") and again on "a block wider than 128 lanes that is not a
tileable 1-D element-wise program" for the non-tileable gather, so the 16-lane fallback ran.

**transpose** (`x.t().contiguous()`, one 16x16 tile a program, 328 instructions a tile in the census of `op.transpose/small`). Each of
the 16 tile columns is loaded by a `v.ld.gather` of 16 words with its own index and mask (a splat, an add, a compare and an and), written
to an accumulator row, read back, and stored with a per-row mask (a splat, a compare, an and and an address). In the single-core RTL it is LSU-port bound (`perf_x_lsu` 24,016 of 71,904 cycles,
`perf_mem_rd_beats` 33,024 for 32,768 elements: one beat an element). On the board the gather is the slow path of the hybrid LSU
(`SLOW_UNITS` = 2 units of `NUM_BEATS` = 4 outstanding beats, `rtl/core/tnca_lsu_hybrid.v`): about 8 word requests in flight, 4 useful bytes
each, against the 73-cycle HBM round trip is 0.44 B/cycle a core (inference from the parameters), which is the order of what `op.transpose/large` shows on the board (98.4 M cycles for 128 MiB moved: 1.37 B/cycle on three
cores) shows. The line path (16 lanes of 4 B = one 64 B request, fast LSU path) was never used for the column read.

**cumsum** (`torch.cumsum(dim=-1)`, Inductor's looped scan, 16 lanes an iteration): `tt.scan` was lowered as the sequential fold of the
combiner body: 16 `v.extract`, 15 dependent `vf.add` (5 cycles each) and 16 `v.insert` per 16 elements, 110 instructions and a 90-cycle
dependent chain an iteration (`perf_issued` 102,056 of 128,232 cycles, `op.cumsum/small`).

None of the three is limited by the hardware port in the form the compiler emitted it: the instruction stream and the per-element
requests are the cost.

## 2. What changed (general mechanisms, default on, no benchmark recognised)

Commits on `R42-copy`: `003697e2` (mechanisms, tests, doc), `d03a3cf1` (transposed load without joins, immediates, assert, `m ^ true`), evidence commit follows.

| File | Change |
|---|---|
| `compiler/python/tnca_cc/frontend.py` | `tt.get_program_id` is non-negative (fact for `_contiguous_div`: `(pid*B + arange) // N` is lane-uniform, `% N` affine); masked broadcast load = one guarded scalar load + splat (`_uniform_sload`, no branch: a block with no active lane reads the base pointer); `tt.scan` of one register in log2(n) `v.shuffle.up` steps (`_scan_steps`); a 16x16 tile read down its columns = 16 contiguous loads + a `v.shuffle.xor` register transpose (`_transposed_rows_load`, `_transpose_registers`), when the mask is the separable bound form Inductor writes (`_tile_full_flag`: one scalar test per index that the tile is whole), the cut tile takes the gather under a branch; a whole tile is stored with 16 unmasked stores (`_masked_rows_store`), `_branch_arm` keeps arm-local caches from leaking past the join |
| `compiler/python/tnca_cc/wide.py` | a 1-D 64-bit block that is lane-invariant is one 64-bit scalar (`_form`, `_uniform`): loads from a lane-invariant address (`_load`), `cmpi` (one scalar compare), `lane()`; `_form_pair` builds the (lo, hi) vectors of `invariant + sext(lane)` with vector arithmetic (dead when only an address reads them); `_track_form`/`_rebase` run in every kernel (before: only `index64` kernels), so pointer + (scalar + lane) stays a contiguous access with a scalar base; `_uop` takes immediates only when they fit 16 bits |
| `compiler/python/tnca_cc/optimization.py`, `python/tnca/aot.py` | dead-pure removal on flat control flow for every kernel (was: only tiled loops), `vselect`/`pbin`/`pun` count as pure |
| `compiler/python/tnca_cc/codegen.py` | `assign` into a value no op defined first allocates it (the join of a branch) |
| `compiler/python/tnca_cc/register_blocks.py` | `tt.assert` is allowed inside register chunks |
| `python/tnca/inductor.py`, `doc/compiler_overview.md` | `TNCA_POINTWISE_MID` (default 128): a program the tile planner refuses is lowered as 128-lane register chunks before the 16-lane fallback |
| `compiler/test/test_copy_lowering.py` | unit tests (eager builder): transpose butterfly, log-step scan (also a non-commutative combiner), 64-bit pair arithmetic |

Numerics: embedding, transpose and copy bit-exact (error 0.0 in every run below); cumsum associates in a tree instead of left to right
(tolerance 1e-3 of the point; measured 1.3e-7 to 1.4e-7).

## 3. Results

All tables are produced by `python3 evidence/r42/copy/tables.py [census|prediction]` from `records/`; the full outputs are `tables.txt`,
`census.txt` and `prediction.txt`.

### 3.1 Single-core RTL screens (rtl; base = R42 head `02b7578e`, unmodified tree, same `software_tree` hash in `base0` and `base_tiny`)

| point | base cycles | new cycles | base/new | issued base -> new | error |
|---|---:|---:|---:|---|---|
| `op.embedding/tiny` | 46,381 | 4,248 | 10.92 | 44,088 -> 3,040 | 0.0 |
| `op.embedding/small` | 366,173 | 23,134 | 15.83 | 347,704 -> 14,852 | 0.0 |
| `op.cumsum/tiny` | 2,218 | 1,075 | 2.06 | 1,712 -> 668 | 1.3e-7 |
| `op.cumsum/small` | 128,229 | 57,148 | 2.24 | 102,056 -> 32,948 | 1.4e-7 |
| `op.cumsum/medium` | 256,325 | 114,220 | 2.24 | 204,072 -> 65,844 | 1.4e-7 |
| `op.transpose/tiny` | 2,310 | 2,268 | 1.02 | 1,312 -> 908 | 0.0 |
| `op.transpose/small` | 72,000 | 73,664 | 0.977 | 41,696 -> 28,224 | 0.0 |
| `op.transpose/medium` | 89,840 | 90,944 | 0.988 | 52,096 -> 34,880 | 0.0 |

`op.transpose/small` is 2.3 % slower in this screen (the gate is 2 %; `op.transpose/medium` 1.2 %): the single-core RTL port moves a 64-byte access in about 8 cycles (4 beats of
16 bytes, `perf_mem_rd_beats` 16,512 for 128 tiles of 16 + 16 accesses), the transpose is memory-port bound there (eta 0.42-0.45 before and after;
`perf_x_lsu` 38,784 of 77,968 cycles in the first version), and the register butterfly (128 instructions a tile) is issued in the cycles the port is
idle. The instruction count falls by 32 % (41,696 -> 28,224); the cycles do not, because this screen's memory has no latency and a 8 byte/cycle
port. The reason the board baseline is 17x off the bound (section 1) is not visible here. Not a result for the board.

`op.embedding/medium` (new only; its base RTL run is 40 minutes, the TLM table below carries its ratio): 92,446 cycles, error 0.0.

**Tiny tier, 27 points** (`records/base_tiny.jsonl` = unmodified R42 head vs `records/final_tiny.jsonl` = the final tree, commit `d03a3cf1`): 27/27 pass numerically,
geometric mean base/new 1.151, no point more than 2 % slower (slowest: `model.resnet_block/tiny` 0.983, `op.matmul/tiny/fp32` 0.991, `model.mlp/tiny` 0.997).
Gains beyond the three families: `model.llama_prefill/tiny` 1.214, `model.llama_decode/tiny` 1.157 (their embedding kernels), `op.linear/tiny` 1.134,
`op.rope/tiny` 1.104, `op.attention_prefill/tiny` 1.056, `op.attention_decode/tiny` 1.036. The first candidate tree (`records/cand_tiny.jsonl`, commit `003697e2`) gave
27/27 pass and 1.147.

GEMM epilogue and conv screens (the whole-tile store changes every masked 16x16 store): `op.linear/small` 237,890 -> 226,665 (1.050), `op.matmul/small`
57,824 -> 56,139 (1.030), `op.gemv/small` 107,642 -> 106,818 (1.008; M = 1, the tile is never whole, the cut arm runs). `op.conv2d/small` (its kernel takes the register-transposed load): 560,278 -> 540,703 (1.036), error 4.4e-7 (tolerance of the point), issued 281,042 -> 278,240.

Extra shapes (not benchmark points, `records/ah_all.jsonl`, `records/ah_nomask.jsonl`), all pass: transpose 40x24 (cut tiles, the gather arm), 64x48, 256x128
(error 0.0), embedding v100/d96/t7 and v300/d160/t20 (error 0.0), cumsum 5x37 and 3x100 (errors 9.6e-8, 8.8e-8); the straight-line transpose (no mask, the
form of `op.transpose/large` and `hbm`, whose kernels carry no mask) checked with a GB10-only hook on 64x48 and 128x128 (error 0.0).

### 3.2 SystemC/TLM model of the three-core image (tlm; calibrated config, same compiler on both sides)

The model/board ratios of the base kernels in `R42-kfuse-speed` are embedding 0.87, cumsum 0.79, transpose 0.55 (it understates the board's gather cost), so the
base side of the transposes is the least reliable number below.

| point | base cycles | new cycles | base/new |
|---|---:|---:|---:|
| `op.embedding/tiny` | 17,259 | 1,756 | 9.83 |
| `op.embedding/small` | 123,446 | 6,646 | 18.57 |
| `op.embedding/medium` | 500,891 | 27,107 | 18.48 |
| `op.embedding/large` | 16,144,497 | 859,554 | 18.78 |
| `op.embedding/hbm` | 258,302,230 | 13,743,159 | 18.80 |
| `op.transpose/tiny` | 1,298 | 722 | 1.80 |
| `op.transpose/small` | 49,933 | 14,984 | 3.33 |
| `op.transpose/medium` | 62,721 | 20,086 | 3.12 |
| `op.transpose/large` | 39,248,846 | 8,839,670 | 4.44 |
| `op.cumsum/tiny` | 1,799 | 915 | 1.97 |
| `op.cumsum/small` | 40,913 | 21,464 | 1.91 |
| `op.cumsum/medium` | 75,349 | 39,532 | 1.91 |

Applied to the `ab-int2` board cycles of these 13 points (`prediction.txt`; `op.transpose/hbm` takes the ratio of `large`): effect on the 113-point
geometric mean x1.214 if the board follows the model ratios, x1.102 if only half of each log-gain materialises. A prediction, not a board result:
the final proof is the board A/B.

### 3.3 Static census (every point of the 113 compiled, a fresh compile cache each; instructions of all kernels of the point, both sides of a branch)

`census.txt`: 74 points have different code, 39 are identical. Smaller: embedding (0.37-0.60), cumsum (0.56), llama_decode (0.52-0.69), llama_prefill (0.78-0.88),
attention_prefill (0.85-0.98), rope (0.88-0.99), conv2d/resnet (0.94-0.98). Larger (both arms of the new whole-tile branches): `op.linear` 1.22-1.28,
`op.matmul`/`op.gemv` fp16 1.07-1.14, `model.mlp` 1.11-1.13, `model.gpt2_block` 1.05-1.07, `op.matmul` fp32 1.03-1.06, `op.transpose` 1.40-1.48 (a whole tile runs
the shorter arm: dynamic instruction counts are in the tables above).

### 3.4 Gates

- `make compiler` on GB10 (final tree): compiler-unit PASS (186 tests, 62 skipped), compiler-ir SKIP (`tnca-opt` is not built on GB10, as in every checkout).
- `python3 compiler/test/test_copy_lowering.py`: 5 tests OK.
- ArchBench tiny tier on the single-core simulator: 27/27 numerically pass on the first candidate tree and on the final tree (S_geo 1.147 and 1.151; none slower by more than 2 %).
- No special-casing: nothing recognises a name, shape or dtype; every mechanism is default on; the one new knob (`TNCA_POINTWISE_MID`, default 128) is documented.
  A tile is cut or whole by one scalar test at run time; a lane-invariant block is recognised from the TTIR (`_form`), not from the kernel.

## 4. What was not done, what failed

- `op.transpose/small` is 2.3 % slower and `op.transpose/tiny` 1.9 % faster in the single-core RTL screen (section 3.1); `op.transpose/medium` is 1.2 % slower (89,840 -> 90,944); the model and the board physics say the line path is
  the right form, the screen has no latency. If the lead prefers the screen's verdict, `_transposed_rows_load` is one function (`frontend.py`) and its call site in `_op_tt_load`.
- First version of the transposed load was 5-8 % slower in the screen (16 `vmovi` and 16 `v.mov` for the join, one address add a row); fixed as described in the commit `d03a3cf1`.
  A value cached inside one arm of a branch (a constant, a column predicate) was used after the join and read garbage on the other arm (error 1.12); `_branch_arm` and the
  `get_mask(0)` before the branch are the fix.
- The widest pointwise config the planner offers (up to 1,024 lanes) is refused for the embedding kernel (not a tileable 1-D element-wise program); the 128-lane candidate
  (`TNCA_POINTWISE_MID`) is what it runs. Its effect on other refused kernels is in the census (rope 0.88-0.99, attention_prefill 0.85-0.98).
- conv2d, resnet_block, rope, softmax: only their static code and the tiny-tier screens were looked at (no lowering change aimed at them beyond the shared ones above).
- The mask-free transpose kernel of `op.transpose/large` and `hbm` has no registered point small enough for the simulator; it is checked with the GB10-only hook (3.1).
- No board run, no new image, no RTL change, no ISA change.

## 5. Remaining ideas, ranked by expected gain (inference from the numbers above, no measurement)

1. **Embedding rows through the transfer engine** (for rows of 256 bytes and more): an id load, a TensorMap row and `t.ld.async`/`t.st.async` instead of 54 instructions and a
   32-byte vector load a 16 elements; also hoisting the id load and the assert out of the chunks of one row. The kernel is now 45-54 instructions for 16 elements. Expected
   2-4x on `large`/`hbm` over this change (5 points: +2-4 % on the geometric mean).
2. **Transposes, two tiles a program** (loads of the next tile issued before the butterfly of this one): the screen shows the port idle during the 128-instruction butterfly;
   about 1.3-1.6x over this change (5 points: +1.5-2 %).
3. **Wide scans** (`tt.scan` of 64 lanes = 4 interleaved in-register scans and 3 carry combines; `inductor._install_narrow_blocks` keeps scans at 16 lanes today):
   the loop is now 31 instructions and an 11-cycle-a-step dependent chain for 16 elements; about 1.5x over this change on 3 points (+1.1 %).
4. The whole-tile branch for `_masked_rows_load` (bias loads of the GEMM epilogue) and the rows of attention: +1-3 % on the GEMM points (`linear`, `mlp`, `gemv`) in the screen.
5. conv2d / resnet_block FP32 on the vector FPU (8 points, eta 0.007-0.08): not examined.
