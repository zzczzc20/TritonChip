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
