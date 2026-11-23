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
