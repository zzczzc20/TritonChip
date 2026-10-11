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

The 128-wide int64 graph uses 14 slots (896 B), 14 static stores and 63
static reloads; the manifest reports 52 VRF registers per role, two roles,
and the same scratch size. These are compiler resource/traffic ledgers;
no HBM bandwidth or performance improvement is claimed. The final RCTX
resource fallback now retries the plain kernel and returns ResourceError
instead of an unbound assembly local.

Multi-row/wide reductions, larger GEMM tiles, bank allocation, full resource
pressure coverage, old-profile contract and frozen performance gates remain
unmeasured. All R40-03 gates remain not-measured.
