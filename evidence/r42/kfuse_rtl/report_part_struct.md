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
