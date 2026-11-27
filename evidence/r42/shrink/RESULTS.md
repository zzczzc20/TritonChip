# R42 shrink: area and clock study (OOC and RTL simulation only; no board, no link, no card)

Branch `R42-shrink` (from `R42` `9dc0712b`), worker `shrink`, 2026-11-27/28. Companion files: `CENSUS.md` (M0), `RESULTS_built.md` (what was built, how verified), raw records under `evidence/r42/shrink/`.
**Domains** (every number carries one): **OOC** = Vivado 2023.2 out-of-context synthesis, and place and route where stated, on `xcu280`, x86 host, ideal clock, every port of the unit registered by a generated wrapper,
image parameters of the run-3 image, hierarchy kept (flatten none); **synthesis** = the area worker's kernel synthesis; **sim** = iverilog simulation; **TLM** = the SystemC model of the dse worker, evaluated with its code as the speed-up of the
145-point wall-time geometric mean against the R41 three-core run-3 image (`S145`, "central"; the model predicts 2.07 for today's hardware with the head software; the board measured 2.03); **derived** = arithmetic on recorded figures.
Nothing here is a routed image or a board number.

## 1. Result in brief

1. **Memory is a small part of the tile.** Array LUTRAM is 17,020 of 253,898 tile LUT (6.7 %); memory-like logic in total is about 26-30 %; about 70 % of the tile is arithmetic and datapath multiplexing (`CENSUS.md`). URAM/BRAM capacity is not the constraint (a 512-bit
   array is 8 blocks whatever its depth); width is.
2. **Built and measured (OOC), zero cycle cost, bit/cycle-exact to the shipped RTL in simulation:** the NUCA slice arrays in URAM (-9.2k LUT per slice at unit level, -8.4k in the tile), the banked VRF response crossbar (-7.4k per core). With the area worker's FMA/MAC/converter back end
   (`FMA_IMPL` 465) the **one-tile OOC synthesis goes 249,921 -> 230,979 LUT (-18,942, -7.6 %)**, 24 URAM and +7.5 RAMB36 per tile; the area worker's kernel-synthesis tile was 253,898 (the same netlist in OOC is 1.6 % smaller), so the kernel-flow equivalent is about 234.7k.
3. **Not enough for more cores.** Two tiles per SLR need a tile of at most 143.2k (4 cores), 140.3k (5), 126.7k (6) synthesis LUT (derived, same floorplan arithmetic as the area report with a +1.5k outlet per extra core in SLR0). 231k is 88k above the 4-core limit. Everything identified, built or estimated, adds up to
   about 195-200k. The blocking units are the matrix lane array (51k), the 16 FMA (24.7k), the converters (15.6k), the vector ALU (21-24k), the home slice (29k after the shrink), LSU (19k), transfer engine (19k), VRF (14k after).
4. **No speed from the shrink itself.** Every built change keeps the cycle behaviour, so the TLM effect of the shrink alone is 1.000. What the freed area (about -19k per tile, 6 % lower SLR0 packing: 95.6 % -> 89.9 %) buys is the area-priced levers of the dse screen: a 4x larger slice (`sets128`, URAM) +4.7 % (full bundle,
   S145 2.170 against 2.072), and, in combination (tiny/small/medium points run, heavy points inherit their family), +31 % for a set of five area-priced levers (section 5).
5. **Clock.** The units measured at 6.67 ns in OOC meet it (VRF 5.88 ns path, 5.56 ns after the crossbar change; FMA, MAC lane from the area worker); the family census of a routed tile at 6.67 ns (section 6) and the image derate of 1.5-2.0 decide what the netlist reaches:
   about 100-110 MHz as it is, 120-125 MHz with the nine pipeline fixes (derived), and no evidence for 150 MHz. TLM: all nine fixes at +1 cycle cost -8.5 % cycles at 100 MHz (S145 1.895 against 2.072) and -2.0 % at 125 MHz; if timing closed, 125 MHz would be S145 2.34 and 150 MHz 2.76.

## 2. Tile LUT after each step (OOC tile synthesis `tnca_tile`, run-3 parameters, tile 0, 10 ns, flatten none)

| Step | `FMA_IMPL` / `SHRINK` | Tile LUT | LUTRAM | FF | RAMB36 / URAM / DSP | Delta | Note |
|---|---|--:|--:|--:|---|--:|---|
| 0 default | 1 / 0 | **249,921** | 17,024 | 69,184 | 32 / 0 / 269 | | reproduces the shipped RTL; kernel synthesis tile 0 of the area worker: 253,898 |
| 1 + area FMA/MAC back end and `tnca_cvt_int` (area worker, bit-identical) | 465 / 0 | not run alone | | | | about -5.0k (area worker, kernel flow, BE 13) to -6.4k (with converter) | its parameter is merged; not re-synthesised alone here (memory pressure on the host) |
| 2 + home slice arrays in URAM | 465 / 10 | not run alone | | | | -8,385 (unit table below) | |
| 3 + VRF response crossbar `XBAR` 2 | 465 / 42 | **230,979** | 8,812 | 69,348 | 39.5 / 24 / 269 | **-18,942 (-7.6 %) against step 0** | unit deltas in the tile: home -8,385, VRF -7,432, vector FPU -4,047, matrix -2,489, vector ALU +2,847 (synthesis variation between runs of the same RTL, section 7), others ~0 |
Steps 1-3 together; the intermediate tile runs were not made. The per-unit deltas of the two tile runs are in `ooc/tile_s0_s3_units.txt`.

## 3. M1: the candidates, measured (OOC unit level, run-3 tile parameters, 10 ns, hierarchy kept)

### (a) NUCA slice data array and the slice's other wide arrays (`tnca_home_pipe`, `DATA_RAM`, `AUX_RAM`)

| Variant | LUT | logic | LUTRAM | FF | RAMB36 | URAM | synth WNS 10 ns | Delta LUT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped (LUTRAM, async read) | 38,455 | 28,561 | 9,894 | 8,960 | 0 | 0 | +5.414 | |
| `DATA_RAM` 1 (RAMB, 15 RAMB18) | 30,835 | 27,385 | 3,450 | 9,281 | 7.5 | 0 | +4.674 | -7,620 (-19.8 %) |
| `DATA_RAM` 2 (URAM, 8) | 31,450 | 27,992 | 3,458 | 9,257 | 0 | 8 | +5.276 | -7,005 |
| `DATA_RAM` 1 + `AUX_RAM` 1 | 29,880 | 28,184 | 1,696 | 9,207 | 22.5 | 0 | +4.732 | -8,575 (-22.3 %) |
| `DATA_RAM` 2 + `AUX_RAM` 2 (24 URAM) | **29,214** | 27,518 | 1,696 | 9,205 | 0 | 24 | | **-9,241 (-24.0 %)** |
FF include about 2.8k of wrapper registers. The first two rows labelled `d1a1`/`d2a2` in the raw directories (`hp_d1a1_syn`, `hp_d2a2_syn`) were synthesised with a wrapper that did not yet pass `AUX_RAM`: they are the data-array-only variants. At 6.67 ns (synthesis estimate) shipped +2.084, URAM +1.897 (WNS after place and route: section 6).
**What changed in the RTL.** `data_mem` read moved to S2: the index S3 will use (hit way, or victim way for a writeback) is known there, so the RAM read is issued one stage early and the line is in the RAM output register when S3 starts; the read has no enable while S3 holds. A **registered bypass of the
write of the same cycle** (512 flip-flops, a 2:1 mux per bit) is needed after the RAM output: a victim line read by S2 in the very cycle S3 merges a store into it (another request of the same set: the arbiter's same-line rule does not cover it) returns the old line otherwise. The first version put the bypass
in front of the RAM output register; the tool then refused to map the array to block RAM (`Infeasible attribute ram_style = block`, array implemented in LUTRAM again), which is why the bypass sits after it. `ms_line` and `tg_wdata` are read at the edge that loads `fu_ms` / `fu_t` (the address is the value being loaded).
No cycle is added anywhere: the TLM effect is 1.000 by construction, and the cycle-exact miter confirms it (section 8). **Latency numbers** (measured in the unisim model, `sim/uram_lat.txt`): URAM288 read = 1 + IREG_PRE + OREG cycles (1, 2, 2, 3); the "3-cycle" figure is the maximum-Fmax setting. 1 is used here.
**Larger slice** (`sets128`: 128 sets x 8 ways = 64 KiB per slice): the line array is 4x deeper in the same 8 URAM (width-bound); the cost is in the tag arrays and the valid/dirty/claim bit vectors (4x, FF and read muxes): `hp_s128_*` synthesis rows in `ooc/ooc_summary_all.txt` when the runs finished. TLM (`sets128`, full bundle): S145 2.170 (+4.7 %). The maintenance scan
(`M_SCAN`, 2 cycles per line over all lines) is 4x longer (2,048 against 512 cycles per maintenance operation); the TLM does not price that.

### (b) accumulator file, (d) MSHR/line buffers/LSU queues, scalar RF: not built (analysis in `CENSUS.md`)

The accumulator file (2,560 LUTRAM + valid logic), the scalar RF (7 read ports), `landing`, `ls_data`, `xb_d` and the descriptor table are all RAM-shaped but their read data is consumed in the cycle of the read by logic inside tightly tuned pipelines (matrix sequencer and `a.vread`; the core's R/X stages; LSU completion); a synchronous read needs
the address a stage earlier and, for the LSU `landing` and the transfer engine `ls_data`, one more cycle per completion (agent notes `census/census_lsu_txf.md`). I did not build them: the cost-benefit (about 4-6k, 3.3k, 0.6k each LUT per tile, estimates) is below the risk of changing those pipelines without the cycle-exact bench the home slice has.

### (c) banked VRF (`tnca_vrf_banked`, `tnca_rf_bank`)

The banks are already RAMB36 (4 x 8); the 21k LUT are periphery. Unit OOC (hierarchy kept):

| Variant | LUT | FF | RAMB36 | Delta LUT | routed WNS 10 ns / 6.67 ns | routed CLB |
|---|--:|--:|--:|--:|---|--:|
| shipped | 21,289 | 10,029 | 32 | | +2.375 / +0.792 | 3,745 / 3,742 |
| `XBAR` 1 (AND-OR over registered one-hot selects) | 15,196 | 10,049 | 32 | -6,093 | | |
| `XBAR` 2 (bank-number 4:1 mux, data not zeroed without a response) | **13,916** | 10,049 | 32 | **-7,373 (-34.6 %)** | **+2.706 / +1.11** | 2,487 / 2,448 |
| `VLD_RAM` (valid bits in the bank RAM, post-reset sweep) | 15,966 | 5,897 | 32 | -5,323 (and -4,132 FF) | | |
| `XBAR` 2 + `VLD_RAM` | 8,580 | 5,917 | 32 | -12,709 | | |
`XBAR` 2 is in the cumulative tile. **`VLD_RAM` is not**: it passes its unit miter but deadlocks the core (a post-reset sweep refuses writes, and `tnca_core` requires the writeback never to be refused: `verify/vld_ram_core_deadlock.txt`). A working version needs the sweep to run while `rst_n` is low with the core's internal reset stretched
until it is done and the launch held: a system decision (the expected saving is the -5.3k above, derived).

### (e) DSP-mapped arithmetic (design and estimate; nothing built)

Recorded anchors: the K-fused E4M3 lane on DSP48E2 (`R42-kfuse-rtl`) is 1,734 LUT / 11 DSP at G = 4 (297 LUT of front end on the unchanged 1,437 LUT back end); the shared lane is 3,203 (synthesis). A lane for the K-fused numerics gate the author accepted would cover (i) E4M3/E5M2 G = 4 (existing front end), (ii) FP16/BF16 G = 2
(11 x 11 products fit one DSP; the 24 + W-bit window needs a coarse shift stage because one DSP aligns only 18 positions) and (iii) the FP32 sequential-FMA contract of ISA 9C.1 unchanged (24 x 24 on two DSP cascaded, straight into the same back end). Estimate: front end 0.6-0.9k + back end 1.44k = 2.0-2.4k per lane against 3.2k: **-12k to -19k per tile**, DSP +100 to +200 per tile (8.2k free), one
new opcode/mode as the numerics decision requires. The FP16 front end has never been built or screened; the figure is a derivation, not a measurement. Other DSP candidates (estimates): the 120 pairwise comparators of the vector ALU (3.3k LUT, each already followed by a flop; +1-2 cycles on integer min/max reductions) -3k; `u_mul` fabric adders -0.4k;
the 49 address adders of the slow LSU unit (2 DSP per 64-bit add, operands between registers) up to -2k. Shifters (valu 3.8k, LSU/txf byte shifters 13k) are not DSP-shaped.

## 4. Predicted effect on the 145-point geometric mean (TLM, S145 central; low-high in brackets)

| Step | Cycle effect | S145 |
|---|---|---|
| today's hardware, head software | | 2.072 (1.853-2.072) |
| 1-3 (shrink: FMA/MAC back end, URAM slice arrays, VRF crossbar) | none: cycle-exact | 2.072 |
| + `sets128` (4x slice, URAM; area paid by the shrink) | fewer misses | **2.170** (1.881-2.190) |
| + four more area-priced levers (`te32` 32 transfer slots, `le2` half-line engine, `hl` hardware loop, `vi1` single-cycle vector issue; the last needs about +14k LUT per core, inference) | | **2.714** (1.951-2.915), tiny/small/medium run, heavy points inherit |
| + clock 125 MHz with the nine fixes (+1 cycle each) | -2.0 % cycles at 125 MHz | 3.227 (2.400-3.433) |
These are model predictions, the last two on a partial bundle (77 light points run directly; the family means stand in for the 68 others): the full-bundle runs are listed in section 9 when finished. Not additive in general; the combinations were run as combinations.

## 5. M2: tile size, cores per SLR, predicted speed

Floorplan arithmetic (`tools/r42/shrink/floorplan.py`, derived): inputs of the area report section C (96.74 % CLB packing, non-core LUT 72.3k / 85.2k / 37.6k, routed = 0.983 x synthesis), +1.5k LUT in SLR0 per core beyond the three (outlet with `AXI_LINE_LUTRAM`; mesh routers beyond three are not costed anywhere).

| Tile (synthesis LUT) | cores that fit | largest tile for N cores |
|---|---|---|
| 253.9k (kernel synthesis, today) / 249.9k (OOC) | 3 | 3 cores 258.0k |
| 231.0k (built, OOC); about 234.7k kernel-equivalent | 3 | 4 cores 143.2k |
| about 215k (everything identified and estimated, memory-only incl. `VLD_RAM`, accumulator, scalar RF) | 3 | 5 cores 140.3k |
| about 195-200k (plus the DSP candidates) | 3 | 6 cores 126.7k |
SLR packing after the built shrink (derived from the routed tile 249.6k -> 230.7k): SLR0 95.6 % -> 89.9 %, SLR1 89.7 % -> 84.6 %, SLR2 87.1 % -> 81.3 %.

TLM for the core counts (what the model gives if the tile did fit; base clock): 4 cores S145 2.406 (1.865-2.553), 5 cores 3.011 (2.044-3.260; modelled as a 3x2 mesh with 5 active cores and 6 home slices, so it also gains slice capacity), 6 cores 3.567 (2.125-3.994). Combined with the clock levers: section 9.
**Which units block, and by how much:** 4 cores need -88k from the built tile (231k), -53k from the all-in estimate; the largest blocks that stay are the matrix lane array 49-51k, FMA 24.7k, converters 15.6k, vector ALU 21-24k, home slice 29k, LSU 19.4k, transfer engine 19.4k, VRF 13.9k, core own 9.9k.
Halving the lane counts would remove about 25k (MAC) + 12k (FMA) and still leave about 160k.

__PENDING__
