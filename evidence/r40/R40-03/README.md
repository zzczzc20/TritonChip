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
pass 20/20. The frozen protocol check passes 33/33. The core-sim O0/O1/O2 artifact contract passes 447/447 byte-identical to
independently rebuilt frozen B39 commit be877793. The older R38 hash table
found on the host does not match that frozen source; it is not used as the
R40 reference. contract_reference.json stores the independently generated
B39 hashes; tools/r40/contract_tests.py verifies its commit and exact case
inventory before comparing current artifacts. It is included in r40-tiles.
Rebuild with the locked Triton environment: archive the B39 commit into a
separate directory, run tools/r37/aot_run.py --source "$TILEBENCH_SOURCE"
--profile core-sim --levels 0 1 2 --compile-only, then run the current
artifact_identity.py with --ref-dir pointing at those generated artifacts.
This is an artifact/ABI comparison and not a performance measurement.
RCTX SPMD/roles suites pass on both v4 and v5, using the old CPU migration
path with CUDA_VISIBLE_DEVICES empty. CUDA/TNCA alternation is checked
separately by the native device suite.

Logical GEMM checkpoint: TNCA_GEMM_TILE selects 16,16,16 (default),
16,32,16 or 16,64,16. Standard native Torch mm/addmm graphs with static
shape guards pass 4/4 for each configuration (12/12), including shapes
17x65x33 and 33x31x65, FP16 inputs and zero resident launch transfers.
The inherited CPU config scaling no longer halves the requested tile.
Provenance confirms the wide tiles; their shared A fragment is packed once
per logical dot, with one/three reuses for 32/64 columns. Reuse is scoped
to that dot, preserving control-flow and iteration lifetimes.
Descriptor chunks preserve extents/strides/padding and shift only the last
coordinate operand, even when row and column coordinates share zero.
Triton-to-RTL checks pass 7/7, including this two-row, 64-column copy.
The harness now also compares post-idle RTL memory, so a kernel's exit
cannot bypass the appended assertions. Restoring the coordinate bug is
detected by the numerical checker. Compiler RTL passes 68/68, including
legal 26-byte row strides and invalid-rank error propagation with untouched
output sentinels. Issued counters use the same running ROI after async faults
as cycle/stall/frontend counters.
The default B39 artifact contract revalidates 447/447; frozen protocol
checks pass 33/33. See gemm16x*_final.log, triton_e2e_gemm_final.log and
contract_gemm_final.log. These groups are included in make r40-tiles.

K32/K64 logical selection, bank allocation, full resource pressure and
frozen performance gates remain unmeasured. All R40-03 gates remain
not-measured.

Bank allocation checkpoint: TNCA_BANK_ALLOC=1 uses the declared banked
profile and its low-bit four-bank contract. A bounded search over free
temporary names scores currently mapped co-read SSA values; windows and
role slices stay reserved. Ownership and lifetime checks are unchanged.
Non-banked profiles refuse the option. The option participates in the
Triton cache key; default off retains the 447/447 B39 artifact contract.

Native Torch triad, LayerNorm and FP32 dot pass 3/3 in each mode. The
verified assembly dump hash matches its artifact, and the physical-map
ledger reproduces provenance exactly. Static extra reads: triad 8 to 0,
LayerNorm 19 to 1, FP32 dot 110 to 110. Moves stay 0/24/0 in both modes.
Diagnostic RTL cycles: 587 to 573, 3437 to 3286, 2384 unchanged. FP32 dot
uses a 42-register prefix versus 41; scratch remains 512 bytes. No gain
is claimed there. These are correctness diagnostics, not frozen subset
performance measurements. Wide 64/128 suites pass 12/12 each, compiler
unit checks 74/74. See bank_*.log and bank_off/on.json.
The off/on checks are included in make r40-tiles; the option stays off
by default and R40-03 remains not accepted.

CI cycle contract migration: R38's historical profile hash no longer
matches the frozen B39 capability configuration. CI run 38139803505
completed two full 149-case numerical runs; only that hash check failed,
with every original cycle value exact. R40 now selects a separate reference
(spec/r40/b39/core_sim_cycles.json), pinning the profile hash computed from
frozen be877793 sources and retaining all 149 R38 values unchanged. The
historical R38 file/check remains available. Any later capability/config
change, nondeterminism or cycle difference still fails. See
ci_b39_profile_migration.log. The two per-case reports are retained by CI.
