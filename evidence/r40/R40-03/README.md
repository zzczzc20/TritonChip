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

2026-10-12 CI configuration audit: R36 CFG-01 mistook the scheduler
model JSON field data[latency] for the separate target latency assumptions.
The audit now examines syntax, retains explicit target-key read checks,
and perturbs the actual target latency fields to verify that the selected
scheduler model is unchanged. CFG-01/02/03 pass; no production identity,
profile, numeric tolerance or frozen cycle is changed by this test fix.
See cfg01_latency_namespace_fix.log.

2026-10-12 K32 checkpoint: native FP16 mm/addmm pass 4/4 each for
16x16x32 and 16x32x32, including 17x65x33 and 33x31x65 tails. Row and
column chunks retain loaded snapshots and pack into the target-declared
K32 MMA layout. The N32 test verifies the actual assembly K encoding
against the executed artifact assembly hash; its A fragment is reused
within each dot. See gemm_k32_n16.log and gemm_k32_n32.log. Default B39
artifacts still match 447/447 (k32_default_contract447.log). BF16 reruns
are pending; K64, wider K32 combinations and frozen speed thresholds
remain open and are not enabled as public GEMM configurations.

2026-10-12 BF16 K32 validation: 16x16x32 and 16x32x32 both pass
4/4 native fullgraph mm/addmm cases, with zero launch transfers and
actual K32 assembly bound to the executed artifact hash. Preserve the
N32 first attempt (600-second host wait) and the successful rerun
(TNCA_HOST_TIMEOUT=172800, unchanged device cycle deadline). These
results close dtype correctness for these two tiles; K64 and frozen
performance thresholds remain pending.

2026-10-12 awaited window reuse checkpoint (default off): single IR role
only, async reads/writes pinned until matching waits, aliases and loop
back edges included. Unsupported conditional completion and shared roles
are refused. Window-free scalar control flow remains supported.
11 safety/ledger tests pass; compiler-unit 85 tests, 5 MLIR tests skipped
on the local host without MLIR tools. CI supplies those tools.

Native mm/addmm at 16x64x32 pass 4/4 tails, actual K32 machine encoding,
zero steady launch transfers, logical window sum 80 -> reserved 32.
window_gemm_n64_k32.log retains the original execution ledger. Its old
explicit-operand counter reports 17 vectors; window_binary_resources6.log
independently decodes the six preserved artifacts and counts 47 per GEMM
including implicit window ranges (two copy kernels have no windows).
Artifacts were not rewritten or re-executed by that audit. New opt-in
sealing counts the union, including reuse/alias deduplication; unit tests
exercise emitted w.def and a sealed TNCA IR artifact was also verified.

window_default_contract447.log confirms default-off B39 447/447 with the
new counter implementation. The stronger public GEMM audit now checks
window definitions directly against its actual artifact and provenance.
K64, other resource lifetimes and frozen performance remain pending.

2026-10-13 K64 resource retile checkpoint: immutable pointer-loaded
row/column chunks split into the largest target-declared K fragment that
leaves private VRF. With window reuse enabled on this 64-register target,
logical 16x16x64 uses two ordered K32 MMA commands into one accumulator.
FP16 mm/addmm 4/4 pass, including both tails, zero steady transfers,
actual machine K/dtype/window definitions and logical tile provenance.
Windows 64 -> reserved 32; complete vector union 44 or 45.
Packed source windows are refused until a snapshot retile is implemented;
inputs are never silently reread. BF16 execution also passes 4/4
(`gemm_k64_retile_bf16.log`), with both mm/addmm tail shapes and the same
actual K32 command/window/register audits. `r40-tiles` runs both dtypes.

Retile unit 4/4 includes ordered accumulation, stable parent cache keys,
packed-input refusal and default-off K64 preservation. Compiler-unit 89
tests (5 local MLIR skips). Default B39 contract remains 447/447.
window_options4.log also proves explicit JIT off wins over environment on,
while ordinary default off remains absent from the AOT semantic options.
These are correctness/resource checks, not frozen GEMM speed acceptance.

2026-10-13 CI fault-injection checkpoint: ERR-05 previously raced the
real PING response overwriting its forged reply before the host read it.
The test now pauses its owned RTL child, observes the actual stale file
through the real reader before resuming, and holds the future forgery
stable until it is rejected. No product transport rule changed.
ci_err05_synchronized.log confirms the stale sequence was observed and
future sequence 108 was refused; the full fault suite passes 7/7 in
ci_err05_fault_suite7.log. This fixes an injection race, not a relaxed gate.

2026-10-13 packed window checkpoint: opt-in TNCA_PACKED_WINDOW_UNPACK=1
uses existing v.get/v.put, scalar masks and shifts to expose FP16/BF16 DMA
snapshot lanes, without rereading source memory or using reserved v.unpack.
The compiler prefers real DMA for supported 16-column descriptor arithmetic.
Unit 2/2 checks both packed row halves and the default refusal boundary;
packed_snapshot2.log executes RTL after overwriting the original source,
preserving exact zeros/signs/Inf/NaN payload bits for both dtypes (2/2).
This IR-builder test is diagnostic, not standard Torch performance evidence.

AOT cache options resolve packed unpack, window reuse and bank allocation
before lookup. Explicit off overrides environment on; default off stays
absent. Bank allocation is now a named public AOT option. Option units 3/3
and default B39 byte contract 447/447 pass. Compiler discovery ran 163 tests:
113 passed, 50 unavailable MLIR/tool checks skipped on this source-only host.
The complete CI environment must still check those tool-dependent paths.
Online SDPA's separate standard Torch checkpoint exercises this capability;
no tile/attention speed threshold or full resource-lifetime acceptance is
claimed here. Reproduce with make r40-packed-window-unpack.
