# R40-03 register-block checkpoint — not accepted

Standard native `torch.compile(..., fullgraph=True)` graphs, core-vrf-sync-v5
RTL on gx10-6646. The cap stays 16 by default; tests use 16/32/64/128 and
assert the artifact's actual wide TTIR lowering, not just the requested cap.
Each group passed 12/12: tails 17/31/33/65/127, signed int64 beyond 32 bits,
fp16/bf16/i8/i16/i32, and strided offset aliases with in-place stores.

Wide one-dimensional SSA values map to distinct VLEN register chunks.
Memory operation order is preserved. Predicate rematerialization uses pure
arithmetic and hardware predicate capacity. A controlled straight-line IR
pass spills long-lived vectors to private per-program scratch; control-flow
spills are explicitly refused until dominance analysis is available.

The 128-wide int64 graph uses 14 slots (896 B), 14 static stores and 42
static reloads after preserving contiguous chunk offsets (63 in the initial run); the manifest reports 52 VRF registers per role, two roles,
and the same scratch size. These are compiler resource/traffic ledgers;
no HBM bandwidth or performance improvement is claimed. The final RCTX
resource fallback now retries the plain kernel and returns ResourceError
instead of an unbound assembly local.

Wide single-row reductions now pass standard Torch sum/amax/RMSNorm/softmax/
LayerNorm graphs at 32/64/128 (20/20 per width). Scalar-carry loops execute
in iteration order, with chunked carried state; stores preserve alias order.
Multi-output body-defined argmax folds pass 4/4 per width (16/32/64/128),
including NaN propagation, first equal maximum, tails and exact indices.
All measured launches have zero resident tensor H2D/D2H bytes. The provenance
records actual reduction widths and kinds. Float LayerNorm uses two-pass
variance here; this is not Welford or multi-row coverage.

Multi-row checkpoint: TNCA_REDUCTION_ROWS independently caps the row block;
default 1 preserves the unit-row path, while larger caps retain its 2-D shape.
The standard Torch five-op reduction suite passes 20/20 at each lane cap
16/32/64 with actual two-row fragments in provenance, tails and zero resident
launch transfers. Native argmax passes 4/4 at 16/32 with two rows; a separate
32-wide two-row test adds 2^40 to indices and passes 4/4, verifying exact high
words and int64 stores. Scalar predicates remain predicates when tile select
broadcasts them. Bounded 2-D int64 arithmetic and memory use a low/high pair
per row with checked address narrowing; reshapes that change the row layout,
64-bit row folds and multi-row scans still refuse explicitly.
Single-row 64/128 argmax revalidated 4/4 each; default single-row reductions
pass 20/20. The frozen protocol check passes 33/33. Old-profile artifact and
RCTX contract checks are running separately, against the actual frozen B39.

Larger GEMM tiles, bank allocation, full resource pressure and frozen
performance gates remain unmeasured. All R40-03 gates remain not-measured.
