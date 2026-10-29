# R41 G2: shared multi-precision matrix MAC lane and FMA area rebuild (DSLX)

Status 2026-10-29: **in progress; everything is default-off and is not instantiated by any image.**
Numbers here come from the runs whose logs sit beside this file. "OOC" means out-of-context place and route of one unit at 10 ns in SLR0 on the x86 host (Vivado 2023.2). That is a screen, not an image.

## 1. What was built

| Piece | Files | Default |
|---|---|---|
| Tree leading-zero count | `xls/tnca_lzc.x` (generic `lzc<N>`) | — |
| FMA normalise with the tree LZC | `xls/tnca_fmad.x` → `rtl/gen/tnca_fmad_norm_xls.v`, `tnca_fmad_l3_xls.v` | `tnca_fp32_fma IMPL=0` |
| Image parameter `FMA_IMPL` | `tnca_multicore_hbm_xo` → … → `tnca_core` → `tnca_vector_fpu` / `tnca_matrix` → `tnca_fpu_multi` → `tnca_fp32_fma` | `0` |
| Shared MAC lane | `xls/tnca_mac.x` → `rtl/gen/tnca_mac_{f1,f2,b0}_xls.v`; wrapper `rtl/core/tnca_mac_lane.v` (cuts `CUTS`) | not instantiated |
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
| `dp4_term_is_v3`: each DSP product equals V3's signed field term, for every E4M3 pair | proved (72 s) |
| `dp4_sum_is_v3` (the four-product form) | no verdict within 900 s; follows from the term lemma, since both sums are plain sums of the four terms with no wrap |

## 3. Simulation (Verilator, GB10; `logs/`)

- **Lane at `CUTS=1110`, against the current lane** (`tnca_fpu_multi` at `7d3b2864`, `verify/mkref.sh`):
  - FP32-mode miter: **210,000,000 vectors, 0 mismatches**. That is 8 classes, the F03 classes plus narrow-format widening, with `fa`/`fb` random.
  - Single products, every FP8 pair in every slot, each checked against the FP32 FMA on the exactly widened pair: DP4 E4M3 is 4 × 65,536 pairs, and DP2 covers all four 8-bit code pairs at 2 × 65,536, with random high bytes. **786,432 vectors, 0 mismatches.**
  - Random single products for all 16 DP2 code pairs: **64,000,000 vectors, 0 mismatches.**
- **Multi-product groups against the exact oracle** (`check_mac.py`, 13 stimulus classes covering cancellation, extremes, specials and tiny values): **26,000,000 groups, 0 mismatches**. In 865,267 of them RO_48 is visible, i.e. the result differs from exact rounding.
- **FMA `IMPL=1`** on the F03 miter (`verify/fma_impl1_miter.sh`), against the pre-F03 generated FMA: **210,000,000 vectors × latencies 2 and 3, 0 mismatches** (`logs/fma_impl1/`).

## 4. Cost (OOC at 10 ns, SLR0, every port registered; `ooc/results/`)

| Unit | LUT | FF | DSP | CLB | WNS | Latency (incl. operand reg) |
|---|---:|---:|---:|---:|---:|---:|
| Today's lane `tnca_fpu_multi` LAT 2 (`mac_ooc_v0`) | 2,733 | 405 | 2 | 464 | +3.165 ns | 3 |
| Shared lane `CUTS=1100` (same latency) | 3,501 | 451 | 8 | 535 | **−2.431 ns** | 3 |
| Shared lane `CUTS=1110` | 3,366 | 513 | 8 | 542 | +0.229 ns | 4 |
| Shared lane `CUTS=1111` | 3,168 | 734 | 8 | 538 | +2.017 ns (worst: widen/decode → F1, 7.96 ns) | 5 |
| FMA alone, `IMPL 0` → `IMPL 1` (earlier unit runs) | 2,245 → 1,473 | 402 → 399 | 2 | 365 → 254 | +3.414 → +3.829 ns | — |

The shared lane is **not yet at the cost target** (≤ today's lane):
- At `CUTS=1110` it costs **+633 LUTs (+23 %), +78 CLBs (+17 %) and +6 DSPs** compared with today's lane, and it adds one cycle of latency.
- Its F1+F2 stage, the DP2 path, is at 9.75 ns. That is far over the ~6.4 ns per-stage aim that the screen-to-image calibration asks for.
- The lane already includes the tree-LZC saving of about −770 LUTs. Without it the packed modes would cost about +1,400 LUTs.
- Synthesis puts 1,583 LUTs in F1, for the products and group sums.

At `CUTS=1111` (two cycles more than today) the lane is +435 LUTs, +74 CLBs, +329 FFs and +6 DSPs. Its worst stage, F1, is 7.96 ns, still above the ~6.4 ns aim.

## 5. Run-2 assessment (2026-10-29)

- **Shared MAC lane: not for run 2.**
  - It misses the cost target and the per-stage timing aim (§4).
  - Integrating it also needs several things that do not exist yet:
    - the new opcode's decode in `tnca_matrix`;
    - a 32-bit B operand per lane, where today it is 16 bits (`b_row[g*16 +: 16]`), which is in the matrix operand-capture path;
    - a deeper `FMA_LAT` in the matrix engine;
    - compiler lowering;
    - the TLM model;
    - every gate.
- **`FMA_IMPL=1` (tree-LZC FMA): a run-2 candidate once screened.**
  - It is bit-exact: proved, and passes the F03 miter at 210M × 2.
  - It saves about 770 LUTs per FMA in the unit screen, and there are 96 FMAs. The kernel-level saving is not measured yet.
  - Still needed: the g2c4 kernel synthesis with its tile-0 screen, plus the core suites and `fma-deep-impl1`.
