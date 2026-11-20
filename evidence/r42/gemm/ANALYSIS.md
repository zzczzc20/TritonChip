# R42 gemm: where a GEMM point's cycles go, and what was changed

Domain of every number here: **single-core RTL simulation** (`tb/device/tnca_device.v`, iverilog), profile
`core-vrf-sync-v5-x8-g2c4-dp`, default knobs, host thor. It is a diagnostic screen, not a board result. The
single-core device of this profile has the word transfer engine (8 bytes a beat); the R41 run-3 board image has the
line-wide engine and three cores, so the transfer/matrix ratio on the board differs.

Baseline tree: R42 head `d88e2aa3` (compiler identical to main `4b61abd8`), mirrored to `~/r42/gemm_out/base0` on
thor. Raw records: `base/*.jsonl` (ArchBench records with `--counters`), `base/*.probe` (engine occupancy and event
logs), kernel dumps `dumps/base_*.s`.

## 1. Method

- `tools/gemm_pt.sh TREE OUT POINT` runs one point (`python -m archbench.run --platform tnca --profile ... --point
  ... --counters`), optionally with a kernel dump (`DUMP=1`) and the probe (`PROBE=1|2`).
- The device's 21 counters (`perf_*`, `doc/architecture_overview.md` sec 8.1) partition cycles into issued / stalled
  (by cause) / front-end idle, but have no "matrix engine busy" term. `tools/gemm_probe.v` is a second simulation
  top (not RTL, not part of the device; built by `tools/gemm_mkprobe.sh`, selected with `TNCA_DEVICE`) that reads
  `dut.u_core.mtx_busy`, `txf_busy_any`, `mtx_cmd_valid`, `txf_is_xfr`, `mtx_done`, `txf_done`, `x_valid` by
  hierarchical reference. Per launch it prints the cycles with the matrix engine busy (and busy with an MMA), the
  transfer engine busy, both, neither; with `+gprobe_ev` one line per command issue/completion and per issued
  instruction. Cycle counts with the probe equal the unprobed device's (6265 for op.linear/tiny both ways).

## 2. Baseline: engine occupancy

| point | cycles | MMAs | MMA-busy cycles | MMA busy | transfer engine busy | both idle | issued instr. | instr./MMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| op.matmul/tiny/fp16-m16-k32-n16 | 1 212 | 2 | 336 | 27.7 % | 40.6 % | 31.7 % | 323 | 162 |
| op.linear/tiny/fp16-m16-k64-n64 | 6 262 | 16 | 2 688 | 42.9 % | 50.3 % | 38.4 % | 1 932 | 121 |
| op.gemv/tiny/fp16-k64-n64 | 8 687 | 16 | 2 696 | 31.0 % | 52.1 % | 29.0 % | 2 356 | 147 |
| model.mlp/tiny, launch 1 (linear + GELU fused epilogue) | 69 290 | 64 | 10 752 | 15.5 % | 16.5 % | 76.5 % | 21 148 | 330 |
| model.mlp/tiny, launch 2 (linear) | 15 622 | 64 | 10 752 | 68.8 % | 73.0 % | 18.8 % | 3 132 | 49 |
| op.matmul/small/fp16-m64-k128-n128 | 60 906 | 256 | 43 008 | 70.6 % | 62.9 % | 19.8 % | 14 204 | 55 |
| op.linear/small/fp16-m64-k256-n256 | 244 321 | 1 024 | 172 032 | 70.4 % | 72.2 % | 19.3 % | 48 254 | 47 |
| op.gemv/small/fp16-k256-n512 | 241 939 | 512 | 86 080 | 35.6 % | 57.3 % | 20.0 % | 51 640 | 101 |

One `m.mma.dp.async.k16.f16` occupies the engine 168 cycles (4096 MACs: 24.4 MAC/cycle against the 32 of the
bound). A 16x16 FP16 tile transfer (512 bytes) occupies the word transfer engine 75 cycles plain and 99 transposed
(`t.ld.async.h.t`, the B operand of a linear layer); a `td.load` snapshot about 100.

## 3. Baseline: the structure of one output tile (op.linear, events in `base/op.linear_tiny*.probe`)

The kernel is one program per 16x16 output tile, two programs at a time on two Role Contexts (RCTX SPMD), each:

```
entry (once a context):  2 TensorMaps built, fence, 2 td.load        557 cycles for both contexts (launch 9400 -> MMAs from 9957)
K step (K/16 times):     arm; t.ld A; wait;  arm; t.ld.t B; wait;  arm; m.mma.dp; wait        20 instructions
epilogue (16 rows):      a.vread; masked v.ld of the bias row; 2 cvt; vf.add; mask (splat, cmp, p.and); masked v.st   14 instr./row
```

- K loop, two contexts interleaved: one MMA every **195 cycles** (issued at 10236, 10406, 10623, 10818, 11013, ...):
  168 of MMA and 27 idle. Per K step the transfer engine is busy 75 + 99 = 174 cycles, so in this device both
  engines are near their limit in the K loop: matrix 86 %, transfer 89 %. Nothing in one context overlaps (every
  command is awaited before the next is issued); the overlap that exists comes from the second context.
- Epilogue: both contexts leave their K loops together and drain together: **1 068 cycles for two tiles**
  (11766 -> 12834), 448 instructions, with both engines idle. 224 instructions a tile; the bias row is re-loaded
  under a different predicate for each of the 16 rows and the row mask is rebuilt per row.
- Split of op.linear/tiny (6 262 cycles, 4 tiles): entry 557 (9 %), K loops 2 x 1 530 (49 %), epilogues 2 x 1 068
  (34 %), program bookkeeping and exit the rest. Of op.linear/small (K = 256): 1 024 x 168 = 172 032 MMA (70 %),
  K-loop idle about 27 x 1 024 = 27.6 k (11 %), epilogues 64 x 534 = 34 k (14 %), the rest entry and bookkeeping.
- Three barriers a program (one per transfer, one per MMA); a context's slice has 4 of the core's 8 when two
  programs share it. `TNCA_GEMM_TILE=16,32,16` needs 5 (three transfers, two MMAs): RCTX SPMD is then refused and
  one context runs everything serially (dump of op.linear/tiny with that tile: barriers `b0..b4`, every command
  awaited). That is a mechanism behind R41's "16,32,16 is slower", independent of the tile shape itself.
- Live registers: 2 windows x 8 vector registers (v48-v63) and v0-v7 of a context's 64; 1 accumulator of a slice's 8.
- The existing `TNCA_SINGLE_ROLE_PIPELINE=1` (same device): op.linear/tiny 6 309 (baseline 6 265), op.matmul/tiny
  1 036 (1 214). It overlaps every second step only and keeps three barriers.

## 4. Baseline: op.gemv (M = 1) is a different kernel

Inductor's template with M = 1 drops the row index of A (`offs_a_m = rm % 1`), so A is one row of K broadcast down
the tile. The importer lowered that as (dump `dumps/base_gemv_tiny.s`, loop `for_5`), **per K step**:
`v.ld` of the row, 16 `v.st` of it into scratch, a TensorMap over the scratch (7 `s.st.d`), `fence`, `td.load`,
then the transfer back. Counters of op.gemv/small: `perf_x_lsu` 68 708 and `perf_x_txf` 71 496 of 241 939 cycles;
the transfer engine is busy 270 cycles per K step against 174 for op.linear, and the matrix engine 35.6 %.
The 16-row epilogue stores 15 rows that are predicated off (M = 1).

## 5. What was changed (both default, no knob needed)

1. **Broadcast-row fragment as one transfer** (`frontend.py` `_broadcast_row_fragment`, called from `_op_tt_load`;
   `_hoisted_tile_descriptor(..., broadcast=True)`). A tile whose row index does not move is a TensorMap with outer
   byte stride 0 (ISA sec 9D.3: `addr = base + sum i_d * byte_stride_d`, signed strides; sec 9E.2 refuses only an
   innermost stride that is not one element). The map is built once at entry; the K position is the column
   coordinate. Applies to any 16-bit tile all of whose readers are `tl.dot` operands; a broadcast tile that
   arithmetic reads keeps its one vector load. Bit-identical results (same fragment contents).
2. **Software pipelining of the transfer/MMA loop** (`tnca_cc/gemm_pipeline.py`, applied in `aot.py` before RCTX
   SPMD; ablation `TNCA_GEMM_PIPELINE=0`). The loop is rotated onto two window sets: the transfers of step i+1 are
   issued while the matrix engine runs step i; all transfers of a step count on one barrier and all matrix commands
   on another (2 barriers a program whatever the tile; before: one per command). Same commands in the same order
   into the same accumulators: bit-identical results. Refused (loop unchanged) for bodies that are not
   `<load part><matrix part>`; falls back to the serial loop when the doubled windows do not fit.

Measured effect: `RESULTS.md`. In this device the pipelined K loop is bounded by the word transfer engine (174 busy
cycles a step): one MMA every ~182 cycles instead of 195. An engine-bound K loop (168) needs fewer transfer bytes
per MMA (register blocking, below).

## 6. Not done, with the measured reason

- **Epilogue** (34 % of op.linear/tiny, 14 % of op.linear/small, both engines idle): 14 instructions a row where
  `a.vread`, add, convert, store are 4-5. Needs the bias row loaded once and the row mask not rebuilt per row
  (the per-row predicated bias load cannot be merged without proving the load safe under the column mask alone).
- **Epilogue under the engines**: two contexts in phase finish their K loops together, so the drain never overlaps
  matrix work. With the pipeline a program holds 2 barriers, so 4 contexts fit the barrier slices
  (`TNCA_RCTX_SPMD_SHARED=4`); not measured.
- **Register blocking in N/M with a cost model**: 16x32 output blocks reuse the A fragment for two MMAs (1.5
  transfers an MMA: 137 transfer-busy cycles a step against 168 of matrix work, so the K loop becomes matrix-bound in
  this device). The pipeline removes the barrier obstacle (sec 3). Two more pieces are needed: issue the next step's
  transfers between the first and second matrix command of a step (as written, a step with n > 1 commands overlaps
  its transfers with the last command only), and choose the block per shape (N <= 16 or an N tail must not pay a
  second, padded MMA): cost(bm, bn) = MMAs(bm, bn) x 168 + max(0, transfers x T - MMAs x 168), from the shape and
  the register budget (2 stages x fragments x 8 registers <= 64 minus temporaries).
- **Entry**: 4 serial `td.load` of ~100 cycles (9 % of op.linear/tiny); the two contexts' maps are identical.
