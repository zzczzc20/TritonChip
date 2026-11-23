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
