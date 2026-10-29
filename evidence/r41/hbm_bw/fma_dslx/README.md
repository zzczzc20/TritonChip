# R41 G2: shared multi-precision matrix MAC lane and FMA area rebuild (DSLX)

Status 2026-10-29: **default-off everywhere; run-2 candidate (see §5). `m.mma.dp.async` is approved by the author (ISA §9C.4).**
Numbers here come from the runs whose logs sit beside this file. "OOC" means out-of-context place and route of one unit at 10 ns in SLR0 on the x86 host (Vivado 2023.2). That is a screen, not an image.

## 1. What was built

| Piece | Files | Default |
|---|---|---|
| Tree leading-zero count | `xls/tnca_lzc.x` (generic `lzc<N>`) | — |
| FMA normalise with the tree LZC | `xls/tnca_fmad.x` → `rtl/gen/tnca_fmad_norm_xls.v`, `tnca_fmad_l3_xls.v` | `tnca_fp32_fma IMPL=0` |
| Image parameter `FMA_IMPL` | `tnca_multicore_hbm_xo` → … → `tnca_core` → `tnca_vector_fpu` / `tnca_matrix` → `tnca_fpu_multi` → `tnca_fp32_fma` | `0` |
| Shared MAC lane | `xls/tnca_mac.x` → `rtl/gen/tnca_mac_{s1,s2,f2,b0}_xls.v` (`f1` = S1+S2 for the unit screens); wrapper `rtl/core/tnca_mac_lane.v` (five cuts, final `0b11101`) | `tnca_matrix MAC_DP=0` |
| Matrix integration | `tnca_matrix` (`MAC_DP`, `MAC_CUTS`, `cmd_dp`, `cmd_bdt`), decoder func 0x02, core capability bit 12, assembler/disassembler/ISA tables | `MAC_DP=0`: ILLEGAL_INSTRUCTION |
| Oracle | `tools/model/fdp.py` `lane_mac_ref` | — |
| ISA proposal | `doc/isa/TNCA-ISA-v0.1.md` §9C.4 `m.mma.dp.async`, **not approved** | — |

The lane modes are:
- **0, FP32:** 1 MAC per cycle. This is `tnca_fpu_multi`, bit for bit; the widening follows `fmt`.
- **1, DP2:** 2 products per cycle. Each operand word holds two 16-bit slots, and each operand can be FP16, BF16, E5M2 or E4M3; an 8-bit element sits in the slot's low byte.
- **2, DP4:** 4 E4M3×E4M3 products per cycle, summed exactly through a 4-DSP fixed-point cascade.

The packed modes compute `C <- RNE_f32(C + RO_48(group sum))`. They feed one 48-bit "product" into the shared, unchanged FP32 back end (align, normalise, round).

E5M2 and mixed FP8 run at 2×, not 4×. A 4-product E5M2 sum needs a 66-bit field plus alignment, which would cost more than the lane itself (see the r41-pmac V1 numbers in `evidence/r41/pmac/lane/README.md`).

## 2. Proofs (formal, Z3 through XLS `prove_quickcheck_main`; `logs/proofs.log`)

| Property | Result |
|---|---|
| `lzc{48,64,89,128}_is_clz`: the tree LZC equals `clz` for every input | proved |
| `normalise_equiv`: `fma32d_normalise == fma32_normalise` for every input | proved (3 s) |
| `fp32_mode_is_fma32`: lane mode 0 equals `tnca_fma::fma32` for every (a, b, c, sub_c, mul_only, ftz) | proved (9 s) |
| `dp2_fp16_bf16_is_v2`: the DP2 front end with FP16/BF16 codes equals the r41-pmac V2 (oracle-verified) | proved |
| `dp4_term_is_v3`: each DSP product (with the folded sign correction) equals V3's signed field term, for every E4M3 pair | proved (9.5 min) |
| `dp2_x_is_ref`: the one-adder DP2 sum equals `tnca_fdp::dp2_align`, every operand pair and code | proved (3 s) |
| `dp4_sum_is_v3` (the four-product form) | no verdict within 900 s; follows from the term lemma, since both sums are plain sums of the four terms with no wrap |

## 3. Simulation (Verilator, GB10; `logs/`)

- **Lane at the final `CUTS=0b11101`, against the current lane** (`tnca_fpu_multi` at `7d3b2864`, `verify/mkref.sh`):
  - FP32-mode miter: **210,000,000 vectors, 0 mismatches**. That is 8 classes, the F03 classes plus narrow-format widening, with `fa`/`fb` random.
  - Single products, every FP8 pair in every slot, each checked against the FP32 FMA on the exactly widened pair: DP4 E4M3 is 4 × 65,536 pairs, and DP2 covers all four 8-bit code pairs at 2 × 65,536, with random high bytes. **786,432 vectors, 0 mismatches.**
  - Random single products for all 16 DP2 code pairs: **64,000,000 vectors, 0 mismatches.**
- **Multi-product groups against the exact oracle** (`check_mac.py`, 13 stimulus classes covering cancellation, extremes, specials and tiny values): **26,000,000 groups, 0 mismatches**. In 865,267 of them RO_48 is visible, i.e. the result differs from exact rounding.
- **FMA `IMPL=1`** on the F03 miter (`verify/fma_impl1_miter.sh`), against the pre-F03 generated FMA: **210,000,000 vectors × latencies 2 and 3, 0 mismatches** (`logs/fma_impl1/`).

## 4. Cost

Unit screens (out-of-context place and route at 10 ns, SLR0, every port registered; `ooc/results/`, per stage in the table):

| Stage (final lane) | Delay | LUT |
|---|---:|---:|
| S1: decode, products, DP4 cascade, DP2 reference choice | 6.17 ns | 689 |
| S2+F2: DP2 shift and one add, \|sum\|, RO_48, mode select | 6.45 ns | 639 |
| B0: align with C | 4.9 ns | 448 |
| norm (tree LZC) | 4.6 ns | 526 |
| round (into the consumer's register) | 5.2 ns | 490 |

Cuts after S1, S2+F2, B0 and norm: result latency 5 against 3 for today's lane (a row returns every 16 cycles, so throughput is unchanged). The first lane (three-adder DP2, cuts 0b1110) had F1+F2 at 9.75 ns and 3,366 LUTs.

Kernel and tile-0 results (g2c4 image; `screens/results/`, RESOURCES.md §10):

| Build | Kernel LUT | Kernel FF | DSP | Tile-0 routed LUT | Tile-0 CLBs | Tile-0 WNS |
|---|---:|---:|---:|---:|---:|---:|
| base | 768,110 | 217,312 | 520 | 239,446 | 40,959 (74.5 %) | +0.728 |
| `FMA_IMPL 1` | 687,182 (**−80,928**) | 217,312 | 520 | 215,094 | 38,228 (69.6 %) | +0.257 |
| `FMA_IMPL 1` + lane | 759,520 (**−8,590**) | 229,885 | 814 | 238,021 | 41,489 (75.5 %) | +0.616 |

## 5. Run-2 assessment (2026-10-29, updated)

- **`FMA_IMPL=1`: go.** Proved and miter-verified bit-exact (210M × 2 latencies, 0 mismatches); `fma-deep`, `fma-deep-impl1`, unit, core and core-g2c4 pass; −80,928 kernel LUTs, −24,352 routed LUTs and −2,731 CLBs in the SLR0 screen. The WNS change (+0.728 → +0.257 ns) is on a core control path, not FMA logic.
- **Shared lane + `m.mma.dp.async` (`MAC_DP=1`, `MAC_CUTS=29`): go from the RTL side, subject to the board image's own timing.** Net −8,590 kernel LUTs against today's machine with +12.6k FFs and +294 DSPs; SLR0 tile-0 CLBs 74.5 % → 75.5 %, WNS +0.616 ns. Verification: 210M FP32 miter, exhaustive FP8 singles, 64M 16-bit singles, 26M oracle groups (all 0 mismatches); `make core-mmadp` and `make core-run2` (G4's run-2 parameter set: `MTX_VRF_REG 1`, `MTX_ABUF 1`, `FMA_IMPL 1`, `MAC_DP 1`, `MAC_CUTS 29`, four contexts, in-order and reordered memory) through the core. Not done: no complete-image bench; no board run; the 113-point compile audit is not run (no compiler code of mine changed).
- **Profile:** `core-vrf-sync-v5-x8-g2c4-dp` (spec/r1_profiles.json) is the profile with the capability; its `MAC_CUTS` is 29 (result latency 5).
- Lowering/TLM contract: `LOWERING_CONTRACT.md`.
