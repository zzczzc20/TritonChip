# R42 K-fused lane: numerics study (model, CPU only)

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Domain: bit-accurate Python model
> (`tools/r42/kfuse_model.py`) on CPU. No RTL, no board, no end-to-end token run. The gate was registered in
> `doc/r42_plan.md` (commit 4b2cd12f) before any result existed; tolerances were not changed.

## What was modelled

One lane step: G exact products plus the fp32 accumulator, aligned to the largest term's exponent, window 24+W bits below it
(W = guard bits), bits below the window dropped by truncation (`trunc`) or round-to-nearest (`rn`), a wide integer sum, ONE RNE
to fp32 per group. Three variants differ in where the accumulator enters:
`acc_in` (accumulator is a term of the window), `acc_after` (group sum rounded to fp32, then an ordinary fp32 add),
`sum_fma` (group sum kept wide, added to the accumulator exactly with one rounding; this is the structure `m.mma.dp` already has).

Validation (`validate_full.log`): G=1 with unbounded W equals the sequential fp32 FMA of ISA sec 9C.1 bit for bit (0 of 32,380
mismatches against the independent oracle); unbounded W equals the exactly rounded group sum (0 of 7,680 per variant); the
numpy model equals the scalar integer model on 401,250 comparisons; the existing E4M3 DP4 lane equals `sum_fma` with unbounded W;
a deliberately mutated model is caught (`mutation_check.log`).

## Study

26 points x 3 seeds x 130 configurations (G in {2,4,8,16}, W in {0,3,8,24}, trunc/rn, 3 variants) at the points' real K
(2048-8192) and their own generators: E4M3 block-wise W8A8 (xl K=2048, 6144), E5M2 and BF16 (format studies), FP16 matmul,
linear, gemv, attention P.V, second projections of mlp/gpt2/llama. Reference: binary64; sequential contract: one RNE fp32 FMA
per k step. The measured error ratio is one realisation of a few hundred rounding flips (the control, sequential in decreasing k,
measures up to 1.48 for FP16 and 1.94 for BF16 against an expected 1.0), so the verdict uses the noise-free expectation of the
same ratio; the measured ratio is reported beside it (`tables.txt`).

## Verdict (registered W in {0,3,8,24})

All 96 configurations per format pass the registered gate, the frozen tolerance holds everywhere, and the lane is not worse
than the sequential contract in any tested configuration except where noted:
- `acc_in` with `trunc` at W=0 for FP16/BF16 passes the gate (expected ratio 1.39 and 1.24) but its accumulator error is 1.9x and
  1.55x the sequential one before output rounding; with W >= 3 it is below 1.0.
- `sum_fma` (the existing lane structure) is at or below 0.99 for every format and G at every W.
- E4M3/E5M2: ratio 1.00 for every variant and W; the group dot product has more accuracy than the sequential chain at G = 16 (0.90).
- Whole-layer emulation of the xl FP8 layers (`layer_tables.txt`): decode and prefill layers pass for every W >= -8 with outputs
  identical to sequential in 3 of 3 seeds for W >= 0; the 28-layer decoder passes all 8 candidates; FP16 attention engines 1.000.

## Minimum viable lane (adder-tree width = 24 + W + ceil(log2 terms) + 1)

| format | registered menu | tree width at G=16 | narrowest window still passing (exploratory, outside the menu) |
|---|---|---|---|
| E4M3, E5M2 | W=0 passes every variant | `sum_fma`/`acc_after` 29 b, `acc_in` 30 b | gate alone W=-4 at G>=8 (25 b); with the dot-level diagnostic W=-2 (27 b); W=-8 fails |
| FP16 | W=0 | 29 b (`acc_in` trunc needs W=3 for the diagnostic: 33 b) | W=-2 only for G<=4; G=8, 16 need W=0 |
| BF16 | W=0 | same as FP16 | W=-2 passes the gate, with the diagnostic G=8, 16 need W=0 |

For comparison the existing exact group-sum field is 38 bits for E4M3 and 64 bits for FP16: a G=16 lane with a 29-bit tree is
narrower than what R41 already built for G=4, and it does not need an exact sum.

## Not covered

End-to-end chosen-token match and the registered chat-quality check (needs the model; after RTL). The two prefill decoders
(`fp8_decoder_prefill`, 40-60 min per candidate). Trained weights and real activation outliers (generators use random weights).
Sticky bits, floor truncation of two's complement values, other tree architectures. NaN/Inf/overflow in the GEMM study (the scalar
model covers them). fp32 matmul and conv (vector FPU). xl `attention_decode` at KV length 8192.

## Files

`tables.txt` (full grids), `layer_tables.txt`, `agg.json`, `layer_agg.json`, `validate_full.log/json`, `mutation_check.py/log`,
raw records `study*.tar.gz`, `layer.tar.gz`, `env.txt`, `commands.txt`; tools `tools/r42/kfuse_*.py`. The xl definitions come from
`origin/R42-xl` at `faf2e236`.
