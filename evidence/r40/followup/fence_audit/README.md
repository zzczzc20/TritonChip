# R40 softmax fence investigation

**Gate delta: none. `R40-01.device-time` remains failed.** This investigation
rules out a proposed compiler change before spending board or synthesis time.
It does not amend the frozen v3 contract or its historical acceptance evidence.

## Reproduce

```sh
python3 tools/r40/followup_softmax_fence_audit.py
```

The command verifies the archived compiler files against `source_manifest.json`,
verifies both original preflight reports against `preflight_manifest.json`, checks
their recorded product identity and matching successful run, and examines the
three frozen softmax kernels. `audit.json` is its captured output.

The files were retrieved read-only from GB10 on 2026-10-26. Original absolute
paths and SHA-256 hashes are preserved. The reports were generated on
2026-10-19 against product tree `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`.
Their configuration includes `TNCA_PIPELINE=2` and `TNCA_MAX_BLOCK=16` on both
paths. These compile-only diagnostics corroborate the barrier's structure; they
are not a new reproduction of the frozen timing experiment. In particular,
`native-stub` uses host storage and never executes a kernel on the FPGA.

## Finding

| Frozen point | Added native fence instructions | Executions per logical row |
|---|---:|---:|
| tiny fp32, 3 × 37 | 1 | 3 |
| small fp16, 64 × 512 | 1 | 32 |
| medium fp16, 80 × 512 | 1 | 32 |

The added `ttg.barrier all` survives as `fence.acq_rel.device` inside the
final reduction loop. The trip counts come from literal loop bounds and the
16-element step; they are static execution counts, not measured cycle overhead.
All three loop bodies reload intermediate results and store normalized results.
The fp32 tiny kernel updates its intermediate buffer in place. The fp16 kernels
read a descriptor-backed fp32 buffer and write an fp16 output buffer.

Small and medium also contain `fence.release.device` before descriptor snapshot
construction in **both** assembly paths. This fence is introduced during lowering
and is absent from the imported TNCA IR. Counting only `tnca.fence()` therefore
misses an ordering instruction. The audit records both counts separately.

## Decision and next experiment

Do not implement straight-line duplicate-fence elimination as the softmax fix:
these fences are separated dynamically by loads and stores. Removing the barrier
or moving it outside the loop requires a separate proof of inter-iteration
dependence, including pointer aliasing, masked tails and Role Context ownership.
The two fp16 pointer arguments have divisibility attributes, not a no-alias
guarantee. The CPU path's omission is not itself proof that omission is safe.

The next useful candidate is a proven loop-barrier transformation or a softmax
lowering that retains the intermediate values without a memory round trip.
Before any board run, compile it with the existing offline preflight, using a
fresh candidate tree identity and output directory:

```sh
python3 tools/r40/v3_compile_preflight.py \
  --source-root "$R40_CANDIDATE" --software-tree "$R40_TREE" \
  --out "$R40_OUT" --device native-stub \
  --only op.softmax/tiny/fp32-r3-c37 \
         op.softmax/small/fp16-r64-c512 \
         op.softmax/medium/fp16-r80-c512
```

Run on GB10 with its Torch/Triton environment. Success only authorizes subsequent
ordering/alias/tail mutation simulation and then the complete native77, HBM14 and
frozen-model campaign on an explicitly allocated board image. Failure rejects the
candidate before hardware measurement. Existing R40 records remain unchanged;
new product identity cannot inherit their measured performance claims.

Numerics, candidate speedup, board state and gate improvement are **not measured**.
