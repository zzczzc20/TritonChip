# R42 K-fused E4M3 lane on DSP48E2: RTL screen (`R42-kfuse-rtl`)

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Every number carries its domain: *RTL sim* (iverilog,
> GB10), *netlist sim* (post-synthesis netlist, iverilog + unisim), *OOC* (out-of-context synthesis, place and route, xcu280, Vivado
> 2023.2 on the x86 host, ideal clock, every port registered), *derived* (arithmetic on OOC and recorded figures). Nothing here is
> an image, a board or a TLM result. All tables are generated from the raw files in this directory by `tools/r42/kfuse_rtl/make_tables.py`.

Task: the author wants the previous FP8 matrix lane compared with a lane that maps the products onto DSP48E2 and fuses K (G products per
cycle, one rounding per group). The numerics study (`evidence/r42/kfuse/REPORT.md`, model, CPU) registered E4M3, G in {2, 4, 8, 16}, W = 0
(window 24 bits below the group's maximum exponent), variant `sum_fma`, `trunc`. This is a screen of that lane in RTL: does it exist, at what
cost, at what clock, and is it bit-exact against the model. No integration into the core, no ISA change, no board, no image.

## 1. Result

- **It exists and is bit-exact to the model (RTL sim).** `rtl/core/tnca_kfuse_lane.v`, G in {2, 4, 8, 16}, W in 0..3: 10,175,920
  vectors over 20 configurations, 0 mismatches (T7). For each (G, W) in {4, 8, 16} x {0, 3}: 1,000,000 random plus about 69,000 directed vectors.
  The 20,000-vector post-synthesis netlist runs (T9) have 0 mismatches. The bench detects 16 of 16 non-equivalent single-point mutants of the RTL (T8).
- **The DSP mapping needs two DSP48E2 per product, not one, and no PCIN cascade for the products** (section 2): the window anchor needs
  the exact msb of each product, which only the product gives, and the truncation is per term. DSP1 forms the 4x4 mantissa product, DSP2
  aligns and truncates it by multiplying with a one-hot power of two (the sign and the truncation rule sit in A and C), a DSP48E2 tree adds the terms.
  DSP48E2 per lane: 3G - 1 (11, 23, 47).
- **Cost, one lane (OOC, 10 ns, T1):** G=4 1,734 LUT, G=8 1,979 LUT, G=16 2,538 LUT, i.e. 434, 247 and 159 LUT per MAC/cycle. The R41 shared lane in the
  same flow: 1,729 LUT for its DP4 path alone (432 per MAC/cycle, 4 DSP) and 3,146 LUT with every mode (786 per MAC/cycle, 8 DSP); recorded 2,792
  (698 per MAC/cycle); V3 recorded 2,929. **At G=4 the new lane costs what the R41 DP4 path costs (+5 LUT, +7 DSP); the gain is all in G.**
  The R41 back end is 1,437 LUT in every lane (80 % of G=4, 53 % of G=16, T4); a product adds 68 LUT (post-synthesis, T4).
- **Clock (OOC, T3):** the G=16 lane places and routes with positive slack at 4 ns (+0.224 ns), 5 ns (+0.895) and 10 ns (+4.346); G=8 at 4 ns (+0.304).
  The limit is the unchanged R41 back end (c to B0, B0 to B1, round to the output register: 3.8-4.1 ns); the new stages need at most 3.2 ns routed. Pipeline depth: 12, 13, 14 cycles for
  G = 4, 8, 16 with default cuts and `IN_REG=1` (11, 12, 13 with `IN_REG=0`); one group per cycle; with `C_LATE=1` the loop a row's accumulator sees is two registers
  plus the combinational round.
- **Per core (derived from OOC, T2, T6):** a 16-lane array is 27,993 LUT / 176 DSP (G=4), 31,868 LUT / 368 DSP (G=8), 40,700 LUT / 752 DSP (G=16), against
  16 x 2,792 = 44,672 LUT / 128 DSP recorded for the shared-lane array: -16,679, -12,804 and -3,972 LUT, at 64, 128 and 256 E4M3 MAC/cycle against 64. It is an
  E4M3-only lane: as a replacement it would drop the FP32, FP16/BF16 and E5M2 modes of the shared lane (not a resource optimisation, discipline 2.5.3). Added beside the
  shared lanes it does not fit any SLR (+31.9k / +40.7k LUT against 3,975 / 22,292 / 28,133 of headroom). Its front end alone on top of the existing back end
  (+9.9k LUT at G=8, +19.1k at G=16) fits SLR1 and SLR2 and not SLR0, where 5,955 (G=8) or 15,091 (G=16) LUT would have to be removed; SLR0 packing would be 98.5 % / 101.2 %.
  With the array as a replacement SLR0 would be 91.8 % (G=8) or 94.4 % (G=16).

## 2. What was built, and why this structure

`rtl/core/tnca_kfuse_lane.v` (hand-written Verilog-2005; the back end is the R41 generated stages, unchanged). One step of a lane
is G E4M3 x E4M3 products plus the binary32 accumulator c, with the arithmetic of `tools/r42/kfuse_model.py`, variant `sum_fma`,
rounding `trunc`, window W (default 0): every non-zero product is aligned to the group's largest leading-one position, 24 + W
bits below it, each term truncated toward zero in magnitude, the terms are added exactly, and the sum is added to c with ONE
round-to-nearest-even to binary32. Interface as `tnca_mac_lane`: `a`, `b` (G bytes each, element k at bits [8k +: 8]), `c`,
`y`, `valid_in`/`valid_out`, parameters `G` (2, 4, 8, 16), `W` (0..3), `CUTS` (optional registers), `IN_REG`, `C_LATE`
(`c` presented with the operands and delayed inside, or late, when the group sum reaches the align stage; the accumulator
loop is then only the back end), `SIGN_MODE`, `TREE`.

Pipeline (G=16, default `CUTS`, `IN_REG=1`): operand registers; decode (mantissas with hidden bit, effective exponents, sign,
NaN and negative-zero flags) into DSP1's A/B registers; DSP1; msb and T; Tmax tree; one-hot shift decode into DSP2's A/B/C
registers; DSP2; four DSP adder levels; |S| and the packed FMA "product"; B0 (align with c); B1 (normalise); B2 (round, combinational
into the consumer's register). Latency 14 (13 with `IN_REG=0`); one group per cycle. `C_LATE=1`: the c to y loop is two registers
plus the combinational round.

**The mapping, per product.**
1. DSP1 (one DSP48E2 per product, 4x4 bits): P1 = ma x mb, exact, 8 bits (hidden bit = exponent != 0).
2. The window anchor is the exact leading-one position of the largest non-zero product: floor(log2 p) = X - 20 + msb(P1), X = ea + eb
   (effective exponents). The msb of a product is msb(ma) + msb(mb) or one more, and which one is only known from the product,
   so the products must exist before anything can be aligned. T = X + msb(P1) (0 for a zero product); Tmax is a LUT compare-select
   tree; L = Tmax - 43 - W.
3. DSP2 (one DSP48E2 per product): A = one-hot 2^sa (26 bits, two's-complement negated, i.e. -2^sa, for a negative product), B = P1
   (pre-shifted left by 8 when the shift is 16 or more), C = 127 for a negative product, so P = +-P1 x 2^sh (+127) is the product
   already on the window's grid, 7 bits finer. P[47:7] is then the truncated term: floor(X/128) for a positive product, and
   floor((127 - X)/128) = -floor(X/128) for a negative one, which is truncation of the magnitude toward zero as the model does it.
   A flushed term (shift < 0) has B = 0.
4. Sum: G-1 DSP48E2 as two-input adders (A:B + C, P register), one per tree node, no LUT (`TREE=1`: the same tree in LUT carry chains).
5. Back end: |S| (up to 29 + W bits) and its sign are packed as the R41 FMA "product" (pm = |S| unnormalised, xpb = Tmax + 257 - W,
   p_zero, p_nan, sign) and go through `tnca_mac_b0_xls`, `tnca_fmad_norm_xls`, `tnca_fma_round_xls`, which are exact for any
   pm and exponent (`xls/tnca_fma.x`, FmaProduct). Specials, signs of zero and exact cancellation therefore follow the same
   rules as the R41 lane, which `kfuse_model` states for `sum_fma`.

**Why not "shifted mantissa x mantissa in one DSP, PCIN cascade".** Two facts of the numerics rule it out for the registered lane:
(i) the anchor needs the exact msb of each product, which a product-free exponent compare cannot give (above), and (ii) the
truncation is per term: sum of floors is not floor of the sum, and a PCIN cascade adds the carries of the dropped bits into
the next term. A single-DSP form with operand shifts and the same C offset would be possible (A can hold ma shifted by up to
22, B mb shifted by up to 13), with the msb carry bit taken from a LUT function of the two mantissas; it was not built. Its
cost is LUTs for two barrel shifters per product, and the DSP count (3G - 1 here) is not what limits the U280 (9,024 DSP48E2,
818 used by the R41 image); the one-hot form trades a DSP for LUTs.

**Inference notes (Vivado 2023.2).** `SIGN_MODE=0` writes the sign as an add/subtract (C -+ A x B); Vivado maps it to two DSP48E2
per product (a multiplier and a separate add/subtract DSP), so the default is `SIGN_MODE=1` (sign in A, one DSP48E2 with C).
The DSP adder tree maps as `C+A:B`, and its second level uses PCIN. DSP1 maps with A/B/M registers (the M register is the P1 register).

## 3. Numerics contract implemented, and the cases the model leaves open

Implemented (all from `kfuse_model.scalar_step`, `sum_fma`, `trunc`, element format E4M3 x E4M3, no `sub_c`):
- products are exact; zero products are dropped and do not set the anchor; the anchor is the largest floor(log2 |p|);
  L = anchor - 23 - W; each term is the magnitude shifted to L with the low bits dropped (toward zero), signed afterwards;
- the group sum is exact (no overflow: |S| < G x 2^(24+W)); then y = RNE_binary32(S x 2^L + c), one rounding, subnormals kept;
- an E4M3 NaN element (0x7F, 0xFF), whatever its partner (also 0): canonical NaN 0x7fc00000; c NaN: 0x7fc00000;
  c = +-Inf: c; an exactly zero total is +0, except that all-negative-zero products with c = -0 give -0 (`zs & sc`);
- c subnormal, c zero, c huge/tiny, exact cancellation: through the same R41 stages as the FP32 FMA.

Left open by the model (listed, implemented as the model computes them):
1. `sub_c` does not exist in the model; the lane has no `sub_c`.
2. A short last group: the model's `scalar_dot` passes fewer elements; a lane gets G elements. Zero padding is exact for every
   result except the sign of an all-zero result: the model's `zs` is taken over the elements it is given, so +0 x +0 padding
   turns an all-negative-zero short group (with c = -0) from -0 into +0. Padding with -0 (e.g. 0x80 x 0x00) reproduces the model.
3. The model does not single out a group whose non-zero products cancel to S = 0 with c = 0: it returns +0 (`total == 0`),
   as the lane does.
4. Overflow to binary32 infinity cannot occur (|S x 2^L| < 2^(L+28+W) is far below the spacing at the largest binary32).
5. `W` is limited to 0..3 by the one-hot port (A is 26 bits) and one 8-bit pre-shift of P1; the registered menu also has W = 8
   and 24, which are not built (W = 8 would need a longer shifter or a third DSP stage for the alignment, W = 24 a 53-bit tree).
6. FP16/BF16 (G = 2, 4): not built. DSP1's product would be 22 bits and the right-shift range 22 + 24 bits, outside what a
   26-bit one-hot port and an 18-bit B port can align; the alignment would be a LUT shifter per product (not measured).

## 4. Verification (domain: RTL simulation, iverilog 12 on GB10, plus a post-synthesis netlist)

- Expected values: `tools/r42/kfuse_model.py` `scalar_step` (Fractions, the definition), configuration `Cfg(G, W, 'trunc', 'sum_fma')`,
  element format E4M3 x E4M3, called by `tools/r42/kfuse_rtl_vectors.py`; nothing in the generator computes a result itself.
- Vectors per (G, W): a fixed directed prefix of 68,7xx-69,0xx vectors, then N random vectors from twelve weighted classes (uniform with and
  without NaN codes, quantised normal operands at random scales, product-exponent spreads including the largest spread and one
  product at the top with the others at or below the window, equal exponents, cancelling pairs with c chosen to cancel, subnormal
  operands, zero and negative-zero products, NaN at any position, truncation-sensitive groups, huge/tiny c), c drawn near the
  group's exponent (+-40), uniform, special, tiny, huge, cancelling. The directed prefix is: every (a, b) byte pair alone in
  element 0 or G-1 (65,536 pairs); every exponent field with equal exponents; the largest and smallest exponent spreads; full
  cancellation; NaN and NaN x 0 at every position against finite/NaN/Inf c; all-zero groups with every negative-zero pattern
  against 20 special c values; huge/tiny c; maximum products of both signs. The result mix of a 1,000,000-vector file is in
  `sim/result_classes.log` (for G=16, W=0: 59,707 NaN, 10,579 infinity, 107,053 +0, 5,901 -0, 2,525 subnormal, 318,987 equal to c).
- Result: T7. Every one of the 20 configurations has 0 mismatches. The six primary files (G in {4, 8, 16}, W in {0, 3}) are 1,000,000
  random plus the directed prefix each. Variants run at 200,000 random: W = 1, 2, G = 2, `SIGN_MODE=0`, `TREE=1`, `C_LATE=1`, all
  optional registers on and off, `IN_REG=0`, and idle cycles in the stream (10 %, 30 %, 50 % bubbles with garbage on the
  operands and `valid_in` low). `valid_out` is checked in every cycle, bubbles included.
- The bench is sensitive (T8): 18 single-point changes of the RTL (an off-by-one in the shift, in the exponent, in the C offset,
  the msb encoder, the subnormal exponent, the negative-zero rule, the zero-sum sign, the NaN flag, the abs, the c delay, the
  max tree, the truncation point, the flush gate) are applied one at a time: 16 are detected (4 to 133,000 mismatches in a
  168,714-vector file), the 2 that survive are equivalent by construction and say why. `nz_without_lsb` is detected by 4 vectors
  only: P1 = 1 needs both mantissas equal to 1, i.e. bytes 0x01 or 0x81, and the directed prefix has exactly four such pairs.
- Every subnormal binary32 result in the vector files equals c, and no result overflows with a finite c (`sim/result_classes.log`):
  the back end is never asked for either with a non-zero group sum.
- Post-synthesis (T9): `tnca_kfuse_lane` synthesised out of context in Vivado 2023.2, written as a functional netlist and simulated
  with the unisim models of the primitives it uses (CARRY8, DSP48E2, FDRE, GND, VCC, LUT1-6, LUT6_2, MUXF7, MUXF8) against 20,000
  vectors each of the same files (a quarter from the directed prefix), which checks that the inferred DSP48E2 configurations (A:B + C,
  the C offset, the registers absorbed into the DSPs) are the arithmetic of the RTL.

## 5. Resources and timing (domain: OOC; tables T1-T6b below)

Flow (`tools/r42/kfuse_rtl/ooc.tcl`, `run_ooc.sh`, `run_queue.sh`; x86 host, at most two Vivado jobs at once, six threads each, no card):
`synth_design -mode out_of_context -flatten_hierarchy rebuilt` (T4: `none`), `create_clock` at 10 ns (also 5 ns and 4 ns) on an ideal source, `opt_design`,
`place_design`, `phys_opt_design`, `route_design`, then `report_utilization`, `report_timing_summary` and the worst path into every register. Tops are in
`tools/r42/kfuse_rtl/ooc_tops.v`: every port of the lane or array is a register of the wrapper (the operand latches before it, the accumulator write after it); the
lane is instantiated with `IN_REG=0`, so the wrapper's operand register is the lane's first register, as in the matrix engine. The flip-flop columns include those
wrapper registers (G=16 lane: 322 of 1,689). The R41 comparison lanes (`tnca_mac_lane`, `MAC_CUTS` 29, `IN_REG=0`) are run in the same flow, in this session.

Reading the tables:
- T1/T2 are the routed results; the array rows are 16 lanes with independent `a`, `b` and `c` per lane (no sharing of the broadcast `a`
  decode, which a real engine would share). An array is within +0.2..+0.9 % of 16 x its lane (linearity check under T2), so the R41 rows
  that are 16 x a lane are linear extrapolations, labelled so.
- The R41 lane with `mode` tied to DP4 is the like-for-like comparison: E4M3 x E4M3, four products, no FP32/DP2 logic. The all-modes lane is
  what the core has today; it also carries the FP32, FP16/BF16 and E5M2 modes, which the K-fused lane does not have.
- T3: WNS is the worst setup slack of the routed design at the constraint; the slack-implied period is the constraint minus WNS, an estimate of the
  lane's own limit, not an image clock. All front-end boundaries have at least 1.5 ns of slack at 5 ns.
- T4 comes from a hierarchy-preserving synthesis, so its totals differ from the routed ones (a different stage of the flow and no flattening).
- T5: the alternatives. `SIGN_MODE=0`: +310 LUT and +16 DSP. `TREE=1` (the adder tree in LUT carry chains instead of DSP48E2): +385 LUT and -15 DSP, which is the
  cost of keeping the tree out of the DSPs. `CUTS=0`: the same arithmetic with no optional register: +249 LUT, 9.7 ns slack-implied period, 859 flip-flops (the cause of the LUT
  increase was not examined).
- T6 (derived): `(a)` the new array replaces the shared-lane array; `(b)` it is added beside it; `(c)` only the front end is added and the existing back end
  is shared (the lane's routed LUT minus the back end's synthesis share, times 16; the mode mux in front of B0 is not counted). SLR0 packing is the recorded
  95.56 % plus the LUT change at 6.13 LUT per CLB over the 55,000 CLB that the two recorded packing figures (95.56 %, 96.74 %) and the 649-CLB headroom imply. It
  is arithmetic on recorded figures; a routed image will differ.
- T6b: equal E4M3 MAC rate (64 per cycle) needs 8 lanes of G=8 or 4 lanes of G=16: 15,832 or 10,152 LUT against 27,664 for 16 R41 DP4 lanes, but the engine's 16-column geometry would change
  and no design for that was made here.

## Tables

### T1. One lane, out-of-context place and route at 10 ns (xcu280, Vivado 2023.2, every port registered)

| lane | LUT | FF | DSP48E2 | WNS (ns) | LUT per MAC/cycle | DSP per MAC/cycle | note |
|---|---:|---:|---:|---:|---:|---:|---|
| K-fused E4M3 G=4, W=0 | 1734 | 732 | 11 | +5.291 | 434 | 2.75 | C_LATE=1 |
| K-fused E4M3 G=8, W=0 | 1979 | 1054 | 23 | +5.017 | 247 | 2.88 | C_LATE=1 |
| K-fused E4M3 G=16, W=0 | 2538 | 1689 | 47 | +4.346 | 159 | 2.94 | C_LATE=1 |
| K-fused E4M3 G=16, W=0, c delayed inside | 2537 | 2041 | 47 | +4.768 | 159 | 2.94 | C_LATE=0 |
| R41 shared lane, mode tied to DP4 (E4M3, 4 products) | 1729 | 526 | 4 | +4.057 | 432 | 1.00 | measured here |
| R41 shared lane, all modes (FP32, DP2, DP4) | 3146 | 711 | 8 | +2.433 | 786 | 2.00 | measured here, DP4 counted |
| R41 shared lane, recorded (evidence/r41/hbm_bw/fma_dslx/README.md) | 2792 | - | 8 | - | 698 | 2.00 | recorded, 4 products |
| V3 E4M3 exact 38-bit field, recorded (pmac/README.md) | 2929 | - | 2 | - | 732 | 0.50 | recorded, 4 products |

### T2. A 16-lane array (independent a and b per lane), same flow

| array | LUT | FF | DSP48E2 | WNS (ns) | MAC/cycle | LUT per MAC/cycle | vs 16 x 2,792 (recorded R41 array) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 x K-fused G=4 | 27993 | 11517 | 176 | +4.657 | 64 | 437 | -16679 LUT (-37.3 %) |
| 16 x K-fused G=8 | 31868 | 16654 | 368 | +4.361 | 128 | 249 | -12804 LUT (-28.7 %) |
| 16 x K-fused G=16 | 40700 | 26799 | 752 | +4.465 | 256 | 159 | -3972 LUT (-8.9 %) |
| 16 x R41 shared lane, DP4 tied (16 x lane, linear) | 27664 | 8416 | 64 | - | 64 | 432 | -17008 LUT |
| 16 x R41 shared lane, all modes (16 x lane, linear) | 50336 | 11376 | 128 | - | 64 | 786 | +5664 LUT |
| 16 x R41 shared lane, recorded | 44672 | - | 128 | - | 64 | 698 | 0 |

Linearity check, array against 16 x its lane: G=4: array 27993 LUT, 16 x lane 27744 LUT (+0.90 %); G=8: array 31868 LUT, 16 x lane 31664 LUT (+0.64 %); G=16: array 40700 LUT, 16 x lane 40608 LUT (+0.23 %).

### T3. Clock: G=16 lane, place and route at three periods (OOC, ideal clock, no SLR crossing)

| period (ns) | WNS (ns) | slack-implied period (ns) | critical endpoint |
|---:|---:|---:|---|
| 10 (p2_g16) | +4.346 | 5.654 | `g_b1.q_reg[23] -> y_q_reg[14]` |
| 5 (p2_g16_t5) | +0.895 | 4.105 | `g_b0.q_reg[70] -> u_lane/g_b1.q_reg[7]` |
| 4 (p2_g16_t4) | +0.224 | 3.776 | `w_c_reg[26] -> u_lane/g_b0.q_reg[21]` |
| 4 (p2_g8_t4) | +0.304 | 3.696 | `g_b0.q_reg[76] -> u_lane/g_b1.q_reg[14]` |

Worst slack into each pipeline boundary of the G=16 lane (ns), routed:

| boundary | at 10 ns | at 5 ns |
|---|---:|---:|
| c0 decode regs | 8.392 | 3.453 |
| DSP1 P | 8.452 | 3.452 |
| T regs | 7.784 | 2.777 |
| Tmax tree | 6.324 | 1.784 |
| DSP2 operands/P | 6.115 | 1.521 |
| adder tree | 7.921 | 2.986 |
| pkt reg | 8.217 | 2.955 |
| B0 reg | 5.340 | 1.015 |
| B1 reg | 4.794 | 0.895 |
| y_q | 4.346 | 1.073 |

### T4. Where the LUTs are (post-synthesis, hierarchy kept, C_LATE=1)

| G | lane LUT (synth) | R41 back end (B0+B1+B2) | per product (decode, DSP1/DSP2 support, T, shift) | Tmax tree | glue (abs S, pack, flags) |
|---:|---:|---:|---:|---:|---:|
| 4 | 1793 | 1437 (80 %) | 272 (68.0 / product) | 26 | 58 |
| 8 | 2095 | 1438 (69 %) | 544 (68.0 / product) | 51 | 62 |
| 16 | 2707 | 1436 (53 %) | 1088 (68.0 / product) | 117 | 66 |

### T5. Variants of the G=16 lane (place and route at 10 ns)

| variant | LUT | FF | DSP48E2 | WNS (ns) |
|---|---:|---:|---:|---:|
| baseline (SIGN_MODE 1, DSP adder tree, W=0, CUTS 0b111011) | 2538 | 1689 | 47 | +4.346 |
| SIGN_MODE 0 (add/subtract form) | 2848 | 1710 | 63 | +5.038 |
| TREE 1 (LUT adder tree) | 2923 | 2090 | 32 | +5.151 |
| W = 3 | 2542 | 1695 | 47 | +4.928 |
| CUTS = 0 (no optional registers) | 2787 | 859 | 47 | +0.285 |
| C_LATE = 0 (c delayed inside the lane) | 2537 | 2041 | 47 | +4.768 |

### T6. Per-core projection (derived from T1/T2/T4 and the recorded SLR0 figures; not an image)

Constants: current array = 16 x 2792 LUT (recorded); SLR0 packed 95.56 %, routed up to 96.74 %, 6.13 LUT per CLB, 649 CLB of headroom, hence 55000 CLB in SLR0; headroom in LUT: SLR0 3975, SLR1 22292, SLR2 28133 (doc/r42_arch_headroom.md sec 1.4).

| option | G | LUT change per core | DSP change per core | SLR0 packing after | fits SLR1 / SLR2 / SLR0 headroom | LUT to remove in SLR0 to fit |
|---|---:|---:|---:|---:|---|---:|
| (a) E4M3 K-fused array replaces the shared-lane array (FP32, FP16/BF16/E5M2 modes would have to live elsewhere) | 8 | -12804 | +240 | 91.76 % | yes / yes / yes | 0 |
| (b) K-fused array added beside the shared-lane array | 8 | +31868 | +368 | 105.01 % | no / no / no | 27893 |
| (c) K-fused front end only, feeding the shared lane's back end (front-end share of the routed lane x 16; the mode mux in front of B0 is not counted) | 8 | +9930 | +368 | 98.51 % | yes / yes / no | 5955 |
| (a) E4M3 K-fused array replaces the shared-lane array (FP32, FP16/BF16/E5M2 modes would have to live elsewhere) | 16 | -3972 | +624 | 94.38 % | yes / yes / yes | 0 |
| (b) K-fused array added beside the shared-lane array | 16 | +40700 | +752 | 107.63 % | no / no / no | 36725 |
| (c) K-fused front end only, feeding the shared lane's back end (front-end share of the routed lane x 16; the mode mux in front of B0 is not counted) | 16 | +19066 | +752 | 101.22 % | yes / yes / no | 15091 |

### T5b. First RTL (commits a13994fc and f891bab0, same logic: A gated by the shift test; same flow), for reference

| run | LUT | FF | DSP48E2 | WNS (ns) |
|---|---:|---:|---:|---:|
| lane G=4 | 1747 | 668 | 11 | +4.953 |
| lane G=8 | 2005 | 926 | 23 | +5.040 |
| lane G=16 | 2592 | 1433 | 47 | +4.870 |
| lane G=16 at 5 ns | 2603 | 1433 | 47 | +0.704 |
| array 16 x G=4 | 28218 | 10493 | 176 | +4.691 |
| array 16 x G=8 | 32300 | 14606 | 368 | +4.302 |
| array 16 x G=16 | 41562 | 22703 | 752 | +4.311 |

### T6b. LUT and DSP for the same E4M3 MAC rate as today (64 MAC/cycle = 16 lanes x 4), lanes only, linear in lane count

| design | lanes | LUT | DSP48E2 | note |
|---|---:|---:|---:|---|
| R41 shared lane, DP4 tied (measured) | 16 | 27664 | 64 | the DP4 path alone, no FP32/DP2 mode |
| R41 shared lane, recorded | 16 | 44672 | 128 | all modes |
| K-fused G=8 | 8 | 15832 | 184 | the engine geometry (16 columns per row cycle) would change; not designed here |
| K-fused G=16 | 4 | 10152 | 188 | the engine geometry (16 columns per row cycle) would change; not designed here |

### T7. RTL simulation against the model (iverilog 12 on GB10; expected values from kfuse_model.scalar_step)

| run | G | W | variant | vectors | mismatches |
|---|---:|---:|---|---:|---:|
| f_g16allcut | 16 | 0 | CUTS=00111111 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=15 bubbles=10% | 268954 | 0 |
| f_g16clate_mid | 16 | 0 | CUTS=00111111 IN_REG=1 C_LATE=1 SIGN_MODE=1 TREE=0 LAT=15 bubbles=0% | 268954 | 0 |
| f_g16mid_bub | 16 | 0 | CUTS=00111111 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=15 bubbles=30% | 268954 | 0 |
| f_g16sm0 | 16 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=0 TREE=0 LAT=14 bubbles=0% | 268954 | 0 |
| f_g16tree1 | 16 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=1 LAT=14 bubbles=0% | 268954 | 0 |
| f_g16w0 | 16 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=14 bubbles=0% | 1068954 | 0 |
| f_g16w1 | 16 | 1 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=14 bubbles=0% | 268954 | 0 |
| f_g16w2 | 16 | 2 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=14 bubbles=0% | 268954 | 0 |
| f_g16w3 | 16 | 3 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=14 bubbles=0% | 1068954 | 0 |
| f_g2w0 | 2 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=11 bubbles=0% | 267714 | 0 |
| f_g4clate_bub | 4 | 0 | CUTS=00111011 IN_REG=1 C_LATE=1 SIGN_MODE=1 TREE=0 LAT=12 bubbles=50% | 268714 | 0 |
| f_g4w0 | 4 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=12 bubbles=0% | 1068714 | 0 |
| f_g4w1 | 4 | 1 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=12 bubbles=0% | 268714 | 0 |
| f_g4w3 | 4 | 3 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=12 bubbles=0% | 1068714 | 0 |
| f_g8clate | 8 | 0 | CUTS=00111011 IN_REG=1 C_LATE=1 SIGN_MODE=1 TREE=0 LAT=13 bubbles=0% | 268794 | 0 |
| f_g8nocut | 8 | 0 | CUTS=00000000 IN_REG=0 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=7 bubbles=0% | 268794 | 0 |
| f_g8w0 | 8 | 0 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=13 bubbles=0% | 1068794 | 0 |
| f_g8w1 | 8 | 1 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=13 bubbles=0% | 268794 | 0 |
| f_g8w2 | 8 | 2 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=13 bubbles=0% | 268794 | 0 |
| f_g8w3 | 8 | 3 | CUTS=00111011 IN_REG=1 C_LATE=0 SIGN_MODE=1 TREE=0 LAT=13 bubbles=0% | 1068794 | 0 |

Total: 10175920 vectors, 0 mismatches.

### T8. Mutation check of the bench (G=4, W=0, directed prefix + 100,000 random vectors)

```
mutant                   mismatches expected verdict
c_offset_0               7378       KILL     killed
c_offset_126             0          EQUIV    survived (EQUIV: no E4M3 product is 127 mod 128 (127 is prime and above 15x15))
c_offset_125             15         KILL     killed
shift_plus1              123284     KILL     killed
xpb_minus1               122726     KILL     killed
nz_without_lsb           4          KILL     killed
msb_6_as_5               50267      KILL     killed
subnormal_exp_0          15712      KILL     killed
negzero_ignores_sign     892        KILL     killed
hi_threshold_24          0          EQUIV    survived (EQUIV: any threshold from 8 to 24 keeps A below 2^26 and B within 16 bits (one-hot range 0..25 plus the 8-bit pre-shift))
drop_6_bits              132948     KILL     killed
s_zero_never             3984       KILL     killed
c_delay_short            127053     KILL     killed
max_is_min               120999     KILL     killed
nan_dropped              4188       KILL     killed
no_flush_gate            3805       KILL     killed
abs_without_plus1        49207      KILL     killed
sign_of_zero_sum         1553       KILL     killed
unexpected results: 0
```

### T9. Post-synthesis functional netlist (Vivado write_verilog -mode funcsim, unisim models) against the same vectors

| netlist | G | W | vectors | mismatches |
|---|---:|---:|---:|---:|
| g16 | 16 | 0 | 20000 | 0 |
| g4 | 4 | 0 | 20000 | 0 |
| g8w3 | 8 | 3 | 20000 | 0 |


## 6. What failed, what was not done, what is not covered

Failures and surprises, all kept:
1. `SIGN_MODE=0` (sign written as an add/subtract so that the DSP's ALUMODE would carry it) does not map to one DSP48E2 in Vivado
   2023.2: it maps the multiplier and the add/subtract to two (T5: 63 DSP and +310 LUT against 47 DSP for the default). It is
   bit-exact (`f_g16sm0`) and left in the RTL as a parameter. An explicit DSP48E2 instantiation with ALUMODE would need a behavioural
   simulation model of the primitive; it was not built.
2. A first version of the `A` operand gating (a flushed term made zero on A) was replaced by gating B (commit `7cde3e27`): -54 LUT
   per G=16 lane, +256 flip-flops (Vivado stops absorbing DSP2's B register). The first RTL's results are in `ooc_v1/` and T5b;
   its simulations were a matrix that was killed when the RTL changed: 10 runs had finished, all with 0 mismatches, 3 were cut
   (`sim_v1_partial/`); because the matrix straddled the change it is kept but not counted, and the final matrix was run again from scratch.
3. xsim on the x86 host cannot run (`/usr/bin/gcc not found`, no root). The post-synthesis netlist was simulated with iverilog on
   GB10 with the Vivado unisim models of the 13 primitives used. The first G=4 netlist run reported 5 mismatches in the first
   five vectors with `valid_out` low: they were issued inside glbl's 100 ns GSR pulse, which holds every flop at INIT
   (`netlist/g4_run1_gsr_artifact.log`); the bench now waits 250 ns under `NETLIST`, and the rerun is T9.
4. The R41 shared lane measured in this flow (3,146 LUT all modes) is 354 LUT above the recorded 2,792 (a sum of per-stage
   figures); the difference was not reconciled. Both are in T1; the comparison that matters here is the DP4-tied lane (1,729 LUT),
   which is E4M3 x E4M3 only like the new lane.

Not done:
- W = 8 and 24 (registered menu) and FP16/BF16 (section 3, items 5 and 6); `sub_c`; the single-DSP-per-product form and a LUT
  product form (no measurement); an explicit-ALUMODE DSP2; a back end narrower than R41's (see below).
- No integration: no matrix-engine change, no ISA mode, no TLM, no image, no board. The lane needs, beyond what is measured here:
  the B latch widened to 16 x G elements per lane and the A-buffer select widened to G elements (doc/r42_arch_headroom.md sec 1.5),
  the capability bit and the new opcode (registered in doc/r42_plan.md, 2026-11-23), a mux between this front end and the shared back end if
  both live in one lane, the oracle for the group size in the SystemC model, and the command overlap of R42-mtx (a K16 command at G=16 is one group).
- The OOC runs have an ideal clock, no SLR crossing and no neighbouring logic: they are screens. A routed image changes every
  number in T1-T3 (the R41 kernel routed 1.6x its post-synthesis delay estimate at 100 MHz, `flow/synth/vivado_ooc.sh`).
- Numbers for the accuracy gate are in `evidence/r42/kfuse/REPORT.md` (model, CPU); this report does not repeat them and adds none.

## 7. Where the cost is, and what to do next (ranked by expected LUT gain per core)

1. The back end is 53 % of the G=16 lane and 80 % of the G=4 lane (T4) and is the R41 FMA back end sized for a 48-bit product
   and subnormal/overflow results that this lane can never produce (|S x 2^L| < 2^(L+28+W), never a subnormal or an overflow
   unless c passes through). A back end written for a 29..32-bit sum and the binary32 c (near/far alignment, no subnormal
   path) is the largest saving available; it was not built or estimated here, and it must reproduce the same `RO`-free rounding
   (the bench above is the check).
2. 68 LUT per product (T4): about half is the one-hot A decode (a 6-input function per bit because the sign is in A). An explicit
   DSP48E2 with ALUMODE (item 1 of section 6) halves it.
3. The G=16 lane is DSP-heavy (47 per lane, 752 per core) because DSP1 and DSP2 are separate: a single-DSP product with operand
   shifts would halve the product DSPs and cost LUTs; the U280 has 9,024 DSP48E2 and the R41 image uses 818, so this is not binding.

## 8. Files, commands, hashes

| path | what |
|---|---|
| `rtl/core/tnca_kfuse_lane.v` | the lane (plus `tnca_kfuse_prod`, `_maxtree`, `_addlvl`, `_addtree`, `_dly`); Verilog-2005; uses the generated `tnca_mac_b0_xls`, `tnca_fmad_norm_xls`, `tnca_fma_round_xls` unchanged |
| `tools/r42/kfuse_rtl_vectors.py` | vector generator (the model's `scalar_step` is the only source of expected values) |
| `tools/r42/kfuse_rtl/tb_kfuse_lane.v`, `run_sim.sh`, `sim_matrix.sh`, `mutation_check.py`, `result_classes.py` | bench, runners, bench sensitivity, result classes |
| `tools/r42/kfuse_rtl/ooc.tcl`, `ooc_tops.v`, `run_ooc.sh`, `run_queue.sh`, `netlist.tcl`, `run_netlist.sh`, `ooc_summary.py`, `stage_slack.py` | OOC flow and its tops, netlist export |
| `tools/r42/kfuse_rtl/collect.sh`, `make_tables.py`, `build_report.sh` | evidence collection, tables, this report |
| `evidence/r42/kfuse_rtl/ooc/<run>/` | final-RTL Vivado reports (`*.gz`: utilisation, hierarchical utilisation, timing summary, 30 worst paths, worst path into every register, log) |
| `evidence/r42/kfuse_rtl/ooc_v1/<run>/` | the same for the first RTL |
| `evidence/r42/kfuse_rtl/sim/` | per-run logs (`f_*.log`), vector generator logs and sha256 of every vector file, `list_f.txt`, `final_tool_sha256.txt`, mutation check, result classes |
| `evidence/r42/kfuse_rtl/sim_v1_partial/` | the interrupted first matrix (first RTL), kept, not counted |
| `evidence/r42/kfuse_rtl/netlist/` | netlist-simulation logs, utilisation of the synthesised netlists, the first run's GSR artifact |
| `evidence/r42/kfuse_rtl/tables.md`, `commands.txt` | generated tables; every command |

The vector files (up to 90 MB each) are not kept: they are a pure function of `(G, W, N, seed)` and the model, and their sha256 are in `sim/*.sha256`.
Source hashes at the time of the final matrix: `sim/final_tool_sha256.txt` (the bench used for the matrix is `c3ef2fe7...`; the bench in the tree adds only a `ifdef NETLIST` delay, used by the netlist runs).
Environment: iverilog 12.0 (GB10, aarch64), Vivado 2023.2 (x86 host, Ubuntu 24.04 WSL2, 16 cores).
