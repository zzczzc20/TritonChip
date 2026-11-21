# Where the cycles of the reduction and normalisation kernels go

Domains are stated per number. **RTL** = single-core RTL simulation, profile `core-vrf-sync-v5-x8-g2c4-dp`,
default knobs, device cycles of the serial grid (a screen, not a board result). **code** = counted from the
assembly the compiler dumped (`TNCA_KERNEL_DUMP=1`) with `loop_census.py` (`census_base.txt` = unchanged main,
`census_head.txt` = this branch). **model** = the scheduler's machine model `spec/r39_sched_model.json`
(`banked2`), itself measured on RTL.

Every point of these families is ONE Inductor kernel (`triton_red_fused_*`), launched once with one program per
row (`XBLOCK = 1`); the programs run on the core's role contexts (`__rctx_program` loop). The per-launch cycles
(`time.per_launch`) are therefore the point's cycles.

## 1. The machine facts that decide (model)

- One instruction issues per cycle, in order; an instruction waits at issue for a register still being written.
- A vector load or store holds the load/store unit 10 + bytes/8 cycles (14 for 16 fp16 lanes, 18 for 16 fp32
  lanes), two in flight. Floating point 5, integer vector 3, SFU (`exp2`, `rcp`, `rsqrt`) 28 cycles.
- So every instruction of an inner loop costs an issue slot, every FP/SFU instruction a latency, and the floor of
  a loop is its memory accesses: about 7 cycles per 16 fp16 lanes per access.

## 2. Unchanged main: instructions per 16 lanes of each inner loop (code) and cycles (RTL)

| Kernel (small point) | Inner loops: instructions per iteration of 16 lanes | of which on data (load/store/FP) | RTL cycles | cycles/element |
|---|---|---:|---:|---:|
| softmax 64x512 | max 24, exp+sum 41, normalise 49 | 3 + 11 + 4 | 449,972 | 13.7 |
| layernorm 64x256 | mean 24, variance 28, normalise 60 | 3 + 5 + 12 | 178,212 | 10.9 |
| rmsnorm 64x256 | mean of squares 25, normalise 56 | 4 + 7 | 138,891 | 8.5 |
| argmax 8x2048 | 44 (+340 after the loop, once per row) | about 15 | 63,301 | 3.9 |
| cumsum 64x256 | 120 (serial scan of 16 lanes) | about 100 | 150,012 | 9.2 |

What the surplus is (from `.s`/`.tnir` of the kernels; the same structure in the tiny and small points):

1. **Loop-invariant work rebuilt every iteration.** The importer emits each Triton statement where it stands and
   the passes of `optimization.py` skip any role with a label or a branch. layernorm's last loop recomputes the
   reciprocal of N with its Newton step (1 SFU + 4 FP), the variance's `rsqrt` with its Newton step (1 SFU + 5 FP)
   and three 32-bit constants per 16 lanes; rmsnorm's last loop the same two SFU chains; softmax's last loop the
   reciprocal of the row sum (1 SFU + 4 FP). Two 28-cycle SFU operations per 16 elements for one scalar per row.
2. **Masks that are always true.** Every access is masked with `roffset + arange < rnumel`; with rnumel a multiple
   of the block the mask is all-true but costs a splat, an add, a compare, a `v.movi` for `other=` before each
   masked load and a `v.sel` after each accumulation. Descriptor loads add a four-compare mask and three `p.and`.
3. **Dead lane-index vectors** (`index + row * rnumel`, which nothing reads).
4. **Scalar bookkeeping.** Each address is `base + ((roffset + row*rnumel) << k)`: 3-4 scalar instructions per
   access per iteration; two branches and a `s.movz #1` per iteration (top-tested loop with an unconditional jump
   back); each loop-carried accumulator goes through two `v.mov`.
5. **Fence and TensorMap.** softmax builds a TensorMap for its fp32 scratch row in every program (7 scalar stores,
   `fence.release.device`, `td.load`) and never transfers through it: the descriptor load was lowered to vector
   loads from the map's base. This fence is the one the profile needs before a snapshot (`snapshot_fence`); the
   "extra debug-barrier fence" of doc/r40_report.md is not in the current softmax code. With a wide block the
   map was rebuilt inside the normalise loop, once per 64 lanes.
6. **One 16-lane register per iteration.** The chain load 14, convert 5, accumulate 5 is waited out per
   iteration; with four registers per iteration the loads issue back to back.
7. **Not addressed here.** argmax's 340 instructions after the loop (a lane-by-lane tournament for the first index
   of the maximum, once per row); cumsum's scan (16 extract/splat/add/insert steps per register: the sequential
   order of the sum); softmax's three passes over the row with an fp32 copy in memory (Inductor's looped form);
   one row per program (`TNCA_REDUCTION_ROWS` 1).

## 3. What was changed (general mechanisms, default on)

`tnca_cc/loop_opt.py` (a pass over every kernel with a loop, `TNCA_LOOP_OPT` default 1), register coalescing in
`codegen.py`, and a wider reduction block in `tnca.inductor`:

| Step | Addresses |
|---|---|
| unused TensorMaps are not built | 5 |
| all-true masks of counted loops folded (loads, stores, selects, predicate ANDs) | 2 |
| copies forwarded, dead pure definitions dropped | 3, 4 |
| loop-invariant pure operations hoisted (innermost first), equal definitions merged | 1 |
| linear addresses become pointers; accesses a constant apart share one through `[s, #imm]` | 4 |
| bottom-tested loops; a value is computed into the register its `assign` would move it to | 4 |
| reduction lane block up to 64 lanes (four registers), at least four blocks per row; 16 for scans, sorts, argmax | 6 |

A loop is touched only if it consists of pure register operations, loads and stores, labels, branches and
assignments: loops with transfers, matrix commands, barriers or fences (GEMM, convolution) keep their code. A
kernel that needs a resource fallback with the pass is compiled without it (`opt_loops_refused`). No operation's
result changes. The 64-lane block is the existing register-block lowering (`tnca_cc/register_blocks.py`, R40's
`TNCA_MAX_BLOCK`): four accumulators combined after the loop, so a row sum is associated differently from 16
lanes (as at `TNCA_MAX_BLOCK=64` in R40); every measured point stays inside its frozen tolerance.

## 4. After (code and RTL)

| Kernel (small point) | Inner loops, instructions per iteration (64 lanes; argmax, cumsum 16) | per 16 lanes | RTL cycles | cycles/element | base/head |
|---|---|---:|---:|---:|---:|
| softmax 64x512 | 16, 53, 25 | 23.5 (was 114) | 170,919 | 5.2 | 2.63 |
| layernorm 64x256 | 16, 24, 55 | 23.8 (was 112) | 80,475 | 4.9 | 2.21 |
| rmsnorm 64x256 | 20, 38 | 14.5 (was 81) | 48,284 | 2.9 | 2.88 |
| argmax 8x2048 | 22 (+297) | 22 (was 44) | 37,202 | 2.3 | 1.70 |
| cumsum 64x256 | 98 | 98 (was 120) | 128,229 | 7.8 | 1.17 |

Steps separately (RTL; `C1_*` = the pass with 16-lane blocks, `C1w64_*`/`D1_*` = with 64 lanes): rmsnorm small
138,891 -> 57,130 -> 48,284; softmax small 449,972 -> 250,253 -> 170,919; layernorm small 178,212 -> 84,667 ->
80,475. A 64-lane block on a row of one block is slower than 16 lanes (`B2_tinyred` against `B2w64_tinyred`:
softmax 3x37 1,347 against 2,019, layernorm 4x64 2,060 against 2,376, rmsnorm 4x64 1,375 against 1,663), hence
the four-blocks-per-row rule.

## 5. What remains, by expected gain

1. **Memory accesses per element** (model): softmax still makes 2 loads of the input, 1 fp32 store and 1 fp32
   load of the scratch row and 1 store per element, about 5 accesses x 7-9 cycles per 16 lanes, i.e. about half of
   its 5.2 cycles/element. Keeping the exp values in registers between the last two loops for rows that fit the
   register file (a persistent reduction) removes two of the five.
2. **SFU per element** in softmax's exp loop and in gelu/silu/swiglu (28 cycles latency per 16 lanes).
3. **argmax's epilogue** (297 instructions per row) and **cumsum's scan** (a log-step prefix network changes the
   association order of the sum, so it needs a numerics decision).
4. **`other=` materialisation** on the remaining masked descriptor loads (a `v.movi` per load) under a row
   predicate that cannot be proven at compile time.
5. **Several rows per program** for short rows (attention's softmax, rows of 16-64 elements), where the
   per-program prologue and epilogue dominate.
6. **Tiled element-wise programs** rebuild lane indices and `index < xnumel` masks per tile (8 compares and 8 adds
   per 128 lanes in the fp16 add); that mask depends on the grid and is outside step 2 of section 3.
