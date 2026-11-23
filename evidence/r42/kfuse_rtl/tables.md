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

