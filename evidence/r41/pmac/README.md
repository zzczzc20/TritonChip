# Packed MAC feasibility: FP8 = 2×FP16 = 4×FP32 on the shared FP32 lane

Status: **feasibility draft, 2026-10-22**. Sections 7 and 8 come from `lane/` (prototype and out-of-context synthesis) and `model/` (SystemC what-if); they are filled in when those runs finish. Nothing here is implemented in the core, and no image is built.

## 1. Goal and constraints

The author's direction: NVIDIA-style throughput doubling with shared compute units. One FP32 FMA datapath runs, per cycle, either 1 FP32-class MAC, 2 FP16/BF16 MACs, or 4 FP8 MACs. Hardware, ISA and compiler change together. Feasibility comes first, then implementation.

Constraints this design has to respect:
- **ISA §9C.1.** `m.mma.async` is defined as one RNE_f32 rounding per k, in increasing k. The ISA says a faster reduction order "must be exposed as a different opcode". Packing changes the reduction order, so it is a new opcode. `m.mma.async` keeps its numerics bit for bit.
- **Non-regression.** The 113 original points' default code must stay unchanged (R41 §6). The packed opcode is therefore opt-in until a policy decision makes it a default anywhere.
- **Area.** SLR0's CLB packing is already 95.24 % in the attempt-3 image (`evidence/r41/stage1/regional_preflight.md`), and every core change is replicated in three SLRs.
- **Timing.** The FMA is already a Stage 2 critical family (F08/F09). The packed lane builds on the split FMA of r41s2-fma (`fma32_align` → `fma32_normalise` → `fma32_round`), not on the R40 one.

## 2. The current datapath

| Item | Value | Source |
|---|---|---|
| Matrix unit | 16 MAC lanes, one per output column; k-outer, m-inner; one accumulator row recurs every 16 cycles | `rtl/core/tnca_matrix.v` |
| Lane | `tnca_fpu_multi`: exact XLS widening of FP16/BF16/E5M2/E4M3 to binary32, then the binary32 FMA (24×24 product, 88-bit align/add field, single RNE, subnormals kept) | `rtl/core/tnca_fpu_multi.v`, `xls/tnca_fma.x` |
| Throughput | 1 MAC per lane per cycle in every input format, so FP8 is no faster than FP16 | same |
| Busy time of one command (ABUF on) | 2a + 18K + 6 cycles: FP16 K16 310, FP8 K16 302 | RTL oracle, `evidence/r40/v3/systemc_m2/README.md` |
| Lane cost (B40 kernel synthesis) | 2,617 LUTs, 448 FFs, 2 DSP48E2 | `evidence/r41/stage2/base-544aed8` |
| Matrix unit cost per core | 50,720 LUTs (2,856 LUTRAM), 12,800 FFs, 32 DSPs | same |
| Tile | about 251k LUTs, 68k FFs, 171 DSPs | same |

E4M3 is already a matrix input format. It is widened exactly, so it is supported, but at the FP16 rate.

## 3. ISA: `m.mma.dp.async`

Proposed as `major 0x40, func 0x02`, with the same operands as `m.mma.async` (`Ad, Va, Vb, Sc`):

| `mod` | Meaning |
|---|---|
| `[1:0]` | K16 / K32 / K64 (as `m.mma.async`) |
| `[3:2]` | A format: FP16, BF16, E5M2, E4M3 |
| `[4]` | source-free event (as `m.mma.async`) |
| `[6:5]` | B format, which must be in the same width class as A |
| `[7]` | B layout: 0 = K×16 row-major (as today); 1 = 16×K n-major, the native layout of a `[N, K]` weight (§6.2) |
| `[9:8]` | reserved, zero |

A new capability bit `CAPABILITY0.MATRIX_DP` gates it.

**Semantics.** The group is G = 4 consecutive k for 8-bit inputs and G = 2 for 16-bit inputs. For each group, in increasing order:

```
C ← RNE_f32( C + RO_48( Σ_{k in group} exact(A[m][k]) · exact(B[k][n]) ) )
```

`RO_48` rounds to 48 significant bits with round-to-odd and an unbounded exponent. Properties:
- **Exact for E4M3×E4M3.** Every exact 4-product E4M3 sum fits in 48 bits: products lie in [2^-18, 2^17.6), and a 38-bit fixed-point field holds the sum. So the result is the correctly rounded RNE_f32(C + Σ) exactly. This is the case the FP8 checkpoints use.
- **When the group sum has more than 48 significant bits,** RO_48 is the only intermediate rounding. That happens only for widely spread exponents: E5M2, mixed formats, and 16-bit pairs more than about 26 binades apart. Round-to-odd to p+2 or more bits makes a later RNE to p bits equal the correct rounding of the sum alone (Boldo–Melquiond). C can still cancel the leading bits, so the operation is specified as written, not as "correctly rounded".
- **Specials.** A NaN operand, 0×Inf, or Inf products of opposite sign give NaN. Otherwise Inf follows IEEE addition with C.
- **Deterministic.** It is fully specified, so it is bit-reproducible across implementations, the same as §9C.1. The exact oracle is `tools/model/fdp.py`, using fractions.

It differs from NVIDIA's tensor cores, whose FP8 accumulation keeps only about 14 bits of alignment and truncates (as studied for Hopper). Here the products are exact, there is at most one 48-bit round-to-odd, and there is one final RNE.

## 4. Lane datapath: what is shared

```
           FP32 mode (m.mma via widening, row ops)        DP modes
a,b ──► unpack ──► 24×24 on 2 DSPs ──► (sp, pm48, xp) ◄── RO_48 ◄── exact group sum ◄── G products
                                              │
                                  existing back end, unchanged in function:
                                  align with C ─► add/sub ─► LZC/normalise ─► RNE ─► y
```

- **The back end is shared unchanged.** In every mode it takes one exact-or-round-to-odd 48-bit "product" and C. So FP32 mode, the existing `m.mma` and the row ops are bit-identical by construction. The prototype checks this with a miter.
- **Multipliers.**
  - FP32: 24×24 as now.
  - 16-bit DP2: each of the two DSPs takes one 11×11 (FP16) or 8×8 (BF16) product. The FP32 multiply is re-split as two 24×12 DSP multiplies, adding on the second DSP's C port.
  - FP8 DP4: four 4×4 products. Either LUT multipliers (each is a few LUTs), or two per DSP with the packing (a₁ + a₂·2¹⁶)(b₁ + b₂·2⁸), which keeps a₁b₁ and a₂b₂ in separate fields.
- **New logic.**
  - FP8: four products placed by absolute exponent into a signed fixed-point field, summed exactly. The field is 38 bits for E4M3 only, 52 for mixed, 66 for E5M2.
  - 16-bit: one exponent compare, one alignment of the smaller product with a sticky bit (one shifted term, so the sticky sign is known), and one add/sub with the FMA's borrow trick.
  - Shared: LZC, normalisation to 48 bits, and round-to-odd.
- **Pipelining.** One accumulator row recurs every 16 cycles, so the lane can take two or three more register stages without stalling the sweep. Only the tag pipeline (`PIPE`) and the row ops' latency grow.

Prototype variants (`lane/`):

| Variant | Content |
|---|---|
| V0 | baseline lane (r41s2-fma) |
| V1 | FP8 DP4, all FP8 pairs (66-bit field) |
| V2 | V1 + 16-bit DP2 |
| V3 | E4M3-only DP4 (38-bit field); E5M2 stays at 1× |

## 5. Matrix sequencer and operands

- **A.** Each cycle needs A[m][k..k+G−1]: 4 contiguous bytes at byte (m·K + k)·E. That is one aligned 32-bit word, simpler than today's element select. ABUF already holds K16 fragments: 8 registers for 16-bit inputs, 4 for FP8.
- **B.** B[k..k+G−1][0..15] is exactly one 64-byte vector register per group (G·E = 4 bytes × 16 lanes). Lane n takes bytes n, 16+n, 32+n and 48+n (FP8), or halfwords n and 16+n. This is fixed wiring. The B row latch grows from 256 to 512 bits.
- **Sweep.**
  - K/G groups × 16 m. With ABUF the busy time is about 2a + (K/G)·18 + 6: FP8 K16 86 cycles (3.5× faster than 302), FP16 K16 166 (1.87×).
  - Double-buffering the B latch hides the per-group read: 80 cycles (3.8×) and 152 (2.04×).
  - K64 FP8 needs ABUF of 16 registers, or the compiler emits K32 pairs.

## 6. E4M3 completion

What FP8 support still lacks (from `evidence/r41/fp8/README.md` §"Hardware/ISA semantic mismatches") and how this design closes it:

1. **One MMA format per command.** Today e4m3×e5m2 runs as an FP16 MMA and is refused beyond K16. `mod[6:5]` gives B its own format, and the DP4 unpack handles each operand's format. The same B-format field could also be added to `m.mma.async`, at the cost of one more widen select per lane.
2. **No 8-bit transposed write in the transfer engine.** A `[N, K]` FP8 weight (`w.t()` in a linear) becomes 16 gathers plus 16 byte stores per K step. That makes the FP8 `_scaled_mm` about 1.8× slower than BF16 with a column-major B in the model (70,870 versus 39,771 cycles, 64×96×128).
   - With packing, lane n needs W[n][k..k+3]: 4 **contiguous** bytes of weight row n.
   - The `mod[7]` n-major B layout reads the weight tile as the transfer engine already writes it (row-major [16][K]). It uses a B fragment buffer (BBUF, 4 registers for FP8 K16) and a 4:1 word select per lane.
   - So FP8 linears need no transpose at all. Cost: about 2 Kbit of FFs and about 512 LUTs per core.
   - The alternative is an 8-bit transposed write in `tnca_tensor.v`.
3. **Fine-grained block scales (1×128 activations, 128×128 weights).** The Hub kernel computes `acc += dot(a, b) · a_s · b_s` once per 128-deep K block.
   - In a 16-column tile, b_s is one scalar per K block, and a_s is per row.
   - Today the promotion is per row: `a.vread` + `a.fma.row` with a ping-pong accumulator. That is about 32 instructions per tile per K block: around 8 % of the MMA time at 1×, but around 25–30 % with DP4.
   - A whole-tile accumulator op `a.fma.acc Ad, As, Vrow_scale, Sscale` (Ad[m][n] = fma(As[m][n], Vrow_scale[m]·Sscale, Ad[m][n]), one RNE) would cost about 32 cycles per tile per K block, reading two accumulators per row over two cycles. It is optional, costed separately.
4. **NaN canonicalisation and saturation.** These already match torch 2.14 for every non-NaN input. NaN payloads are an ISA property, handled in the compiler. No change.

## 7. Area and timing (measured: `lane/README.md`, `lane/ooc/projection.json`)

**Verification.** The lane is bit-exact against the Fraction oracle (`tools/model/fdp.py`), with zero mismatches in every check:

| Check | Count |
|---|---|
| FP32-mode miter against the current lane | 40 M vectors per variant per register placement |
| Every FP8 single product | 1,048,576 per variant |
| Random 16-bit single products | 40 M |
| Multi-product groups | 13 M. In 548,182 of them round-to-odd differs from exact rounding, so the check is not vacuous |
| Regenerated FMA, latency 2 and 3, against 0b0a3cc | 210 M × 2 |

The E4M3 exactness claim holds: every product is a multiple of 2⁻¹⁸, and |Σ|·2¹⁸ < 2³⁸.

**Cost.** U280, routed out of context, per lane (V0 = 2,492 LUTs):

| Variant | Registers needed at 10 ns | Lane LUTs | ΔLUT per core / ×3 | SLR0 CLB packing (now 95.3 %) |
|---|---|---|---|---|
| **V3: E4M3×E4M3 at 4×** | 3 (latency 4) | 2,929 (+18 %) | +7.2k / +21.6k | ~97.4 % |
| V1: all FP8 pairs at 4× | 4 (latency 5) | 3,433 (+38 %) | +15.3k / +45.8k | ~99.8 % |
| V2: + FP16/BF16 at 2× | 5 or more | ~4,140 (+66 %) | +26.6k / +79.8k | ~103 % |

"Needed" means no stage slower than V0's worst stage, which is about 6.4 ns OOC routed.
- V3 at today's latency misses that bar by 0.6 ns.
- V2's worst path is the 11×11 product, the alignment shift and the 67-bit sum: 7.8–8.3 ns even with four registers.

**What changed from §4.**
- RO_48 needs no normaliser: an 18-bit leading-zero count and a 0–18 shift are enough, and V3 has no rounding logic at all.
- The DP2 field is 64 bits.
- Vivado maps "LUT" 11×11 multipliers to DSPs by default. DSP sharing saves DSPs, not LUTs, and it is slower.

**Conclusion.** SLR0 area, not the arithmetic, rules out V1 and V2 in the current floorplan.

## 8. End-to-end effect

**Measured in the model** (`model/README.md`; uncalibrated, so only the ratios count). V3 at G=4 cuts the engine's busy time per FP8 MMA from 302 to 87 cycles, yet FP8 W8A8 linears at M=1/128, 2048×6144 and 2560×9216 run only **0.99×–1.04×** faster. For BF16 at G=2 the result is 1.00×–1.04×.

Reasons:
- **Engine stall is a small share.** MMAs are asynchronous and overlap loads, so only 0.1–4.1 % of the time is a context stalled on the engine.
- **FP8 waits on the weight transpose.** There is no 8-bit transposed write, so `W.t()` costs 37–58 LSU instructions per MMA. That makes FP8 1.2–3.6× slower than BF16 at the same shape.
- **BF16 at M=128 waits on transfer-engine admission.** Barrier wait moves into admission wait, and the total does not change.

Earlier context: the R40 attribution already bounds the answer: in the single-core 2048³ FP16 matmul the matrix engine was busy for about 48 % of the device cycles (2,097,152 K16 commands × 310 cycles against 1.36×10⁹ cycles), and the rest was transfer admission and barrier sleep. Packing shortens only the busy part. It pays fully only where the transfer engine and memory keep up. FP8 also halves the bytes per MAC.

## 9. Software

- **Compiler.**
  - Emit `m.mma.dp` for `tl.dot` with FP8 or 16-bit inputs and an FP32 accumulator, when selected.
  - Selection is a public knob. The proposal is `TNCA_MMA_REDUCTION=ordered|grouped`, default `ordered`. Making it the default for FP8 `_scaled_mm` is a separate decision, because torch's own FP8 matmul on CUDA has no reduction-order contract.
  - Mixed-format and n-major B selection in the frontend; K-group alignment, so tails are masked to zero, which is exact.
- **SystemC/TLM model.**
  - Numerics from the `fdp.py` semantics, since functional execution must match the RTL.
  - Busy-time keys (`model/` adds the timing what-if keys).
- **Runtime.** A capability bit. The compiler refuses `grouped` on an image without it.
- **Tests.** An exact oracle; RTL against the oracle; standard Torch `_scaled_mm`, linear and matmul in `grouped` mode against an independent reference with the same group semantics; tolerance against `ordered`.

## 10. Risks

- **SLR0 area.** If V2 costs about 1.5k LUTs per lane, that is about +24k LUTs per core, or about +5.5 % of an SLR, on top of 95 % CLB packing. V3 or V1 alone may be the only variant that fits without moving logic out of SLR0.
- **Timing.** A wider front end and a deeper pipeline on a Stage 2 critical family.
- **Benefit.** If transfers dominate, the end-to-end gain is much smaller than 2× or 4×. FP8 linears in particular are held back by the B transpose today (§6.2).
- **Default numerics.** The `grouped` order changes results relative to `ordered`. Tests need the new oracle, not a looser tolerance.

## 11. Recommendation and implementation plan

1. **Feasibility answer.** Packed MACs that share the FP32 lane are feasible only for E4M3×E4M3 at 4× (V3: +18 % per lane, +21.6k LUTs for the three cores, SLR0 at about 97.4 %, latency +1). FP16/BF16 at 2× and all FP8 pairs do not fit SLR0 or 10 ns in this floorplan.
2. **Do not integrate V3 into the Stage 2 image yet.** It buys at most 4 % end to end today, at real SLR0 and timing risk, and that image has a budget of 5 full runs.
3. **First, the FP8 hardware change that pays.**
   - Add an 8-bit transposed write to the transfer engine (`tnca_tensor.v`, today 16-bit only), so an FP8 `W.t()` operand is one transfer, as it is for FP16.
   - The alternative is the `m.mma.dp` n-major B layout (§6.2), but it only exists together with V3.
   - The transposed write also serves the existing `m.mma.async.e4m3`.
4. **Then re-run the model what-if with that fix applied** (V3 on and off, plus the transfer-engine and home-slice sensitivity configs in `model/configs`). Integrate V3 only if the gain is then material. The ISA encoding (§3), the oracle (`tools/model/fdp.py`) and the verified lane (`lane/`) stay ready for it.
