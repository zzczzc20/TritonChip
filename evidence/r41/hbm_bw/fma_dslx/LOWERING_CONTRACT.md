# m.mma.dp.async: contract for compiler lowering and the TLM model

Status: ISA §9C.4 is **approved by the author (2026-10-29)**. RTL: `tnca_matrix` with `MAC_DP=1` (default 0). All lines below are verified end to end by `make core-mmadp` (15 format and K combinations, 188 checks) against the exact oracle `tools/model/fdp.py` (`mma_dp_ref`).

## Instruction
`m.mma.dp.async[.k16|.k32|.k64][.A[.B]][.sf] Ad, Va, Vb, Sc` (major 0x40, func 0x02), the operands, barrier token and `.sf` source-free event of `m.mma.async`.
- `mod[1:0]` K (3 reserved); `mod[3:2]` A format; `mod[4]` source-free; `mod[6:5]` B format; `mod[9:7]` reserved zero.
- Format codes (both fields): 0 f16, 1 bf16, 2 e5m2, 3 e4m3. In the assembler one dtype suffix names both A and B (`.e4m3`); two give A then B (`.f16.e4m3`).
- Capability: `CAPABILITY0` bit 12 `MATRIX_DP` (`TNCA_CAP_MATRIX_DP`). Where it is clear, the encoding raises ILLEGAL_INSTRUCTION. The formats' own bits (`MATRIX_F16`, `MATRIX_BF16`, `MATRIX_FP8`) still apply.

## Format pairs and group size G
Every one of the 16 (A, B) pairs is legal, A and B independent.

| Pair | G (products summed per step) | Throughput vs `m.mma.async` |
|---|---:|---|
| e4m3 × e4m3 | 4 | 4× fewer cycles per element (B rows are re-read once per 4) |
| every other pair (f16, bf16, e5m2, e4m3 in any combination) | 2 | 2× |

## Semantics (one accumulator element, K a multiple of G; always true for 16/32/64)
```
for g in 0 .. K/G-1, increasing:
    C <- RNE_f32( C + RO_48( sum over j in 0..G-1 of  A[m][g*G+j] * B[g*G+j][n] ) )
```
- Products and the group sum are exact; `RO_48` is round-to-odd to 48 significant bits (the identity for e4m3 × e4m3). One RNE per group, into binary32.
- Specials: a NaN element, 0 × Inf, or Inf products of opposite sign give NaN; otherwise an Inf product follows IEEE addition with C. A zero group sum takes the IEEE sign of a sum of zero products. No flush-to-zero.
- **It is not bit-identical to `m.mma.async`**: that rounds after every product (G = 1). Never substitute one for the other silently. A compiler flag (r41-pmac proposed `TNCA_MMA_REDUCTION=ordered|grouped`, default `ordered`) is the author's decision.
- C starts from whatever the accumulator holds (`m.zero` first, or accumulate onto a previous result), exactly as `m.mma.async`.

## Operand layout
Same as §9C.2, each fragment in its own format's element size:
- A: a flat row-major byte array, `A[m][k]` at element `m*K+k`; `ceil(16*K*bytes/64)` registers from `Va`.
- B: `B[k][n]` at element `k*16+n`; `K*16*bytes/64` registers from `Vb`. B's element size is B's format (an e4m3 A with an f16 B is legal).
- Limits: a context has 64 vector registers, so A and B together must fit: K = 64 is possible only when both are 8-bit; 16-bit operands go to K = 32. The accumulator readout needs one more register (the test uses `v7`).
- K tails: pad with zero elements, which is exact; K must be one of 16/32/64 per command.

## What lowering should do
- Select `m.mma.dp.async` for FP8/FP16/BF16 matmuls with an FP32 accumulator when the target profile reports `MATRIX_DP`; otherwise keep `m.mma.async`.
- Prefer e4m3 × e4m3 for the Hub FP8 kernels (G = 4).
- Profile: the machine with the capability is a new profile (to be named by the lead); `core-vrf-sync-v5-x8-g2c4` has none.

## TLM model
- Per `m.mma.dp.async`, the sweep has K/G steps of 16 cycles (m inner); one B vector read per step; one A word per cycle; result latency = operand register + `popcount(MAC_CUTS)` (4 with the default cuts 0b1110) against 3 for `m.mma.async`. Label any gain from it "model".
- Accumulate hazards and the drain follow the result latency; the 16-cycle row return makes them irrelevant below 16.
