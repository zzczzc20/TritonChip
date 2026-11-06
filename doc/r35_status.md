# Phase 3.5 Status: Reusing Upstream Infrastructure and Delivering Observable Compiler Optimizations

Contract: [phase3_5_tasks_and_acceptance.md](phase3_5_tasks_and_acceptance.md) v1.0.
Sign-off identifier: `R35_COMPILER_ACCEPTED`. **Currently NOT ACCEPTED (one criterion remaining).**

Full gate execution on the locked Linux primary environment (`Linux-6.17.0-1019-aws-x86_64`, Triton 3.8.0) at commit `0e7c34795cd6` on a clean working tree (run ID `1789956010-0e7c34795cd6`, recorded in [`doc/evidence/r35/acceptance-linux.json`](evidence/r35/acceptance-linux.json)):

| Package | Result |
|---|---|
| R35-01 plan / R35-02 upstream / R35-03 layout | 3/3, 3/3, 3/3 |
| R35-04 opt | **2/3 — R35-OPT-02 not satisfied** |
| R35-05 alloc / R35-06 safety / R35-07 perf | 3/3, 3/3, 3/3 |
| R35-08 package / R35-09 release | 3/3, 3/3 |

**26/27.** The verifier independently re-evaluates the same evidence without trusting its `verdict` or `passed` fields, recomputing from the frozen inventory and raw results to reach the same conclusion: all 27 criteria are present, summaries match raw results line by line, gains recomputed from **raw cycles** yield 3 winners and 1 LL attribution, 0 regressions, and total cycles drop from 5097 to 4811, with only R35-OPT-02 remaining open.

Precondition satisfied: R3 v2 signed off `R3_V2_CORE_VRF_ACCEPTED` at commit `33dbb375828e` on locked Linux (see [`doc/evidence/r3v2/acceptance-linux.json`](evidence/r3v2/acceptance-linux.json)). All primary gains for this phase are measured on its accepted `core-vrf-sync-v1` profile.

## G0 First Step: Demonstrating That Opportunity Exists

Section 6.3 explicitly states that 5% and 3% are **phase targets to be met, not measured speedups**. Before promising any percentage, `tools/r35/scout.py` reads the **machine code currently emitted** by the product compiler, counts by name instructions that can be eliminated, and specifies the removal mechanism and current blocking factors for each.

On `core-vrf-sync-v1`, across the four frozen workloads that compile:

| Workload | Emitted Words | Opportunity | Sites | Possible Savings |
|---|---:|---|---:|---:|
| L01 masked AXPY | 30 | Wide constants | 1 | 3 |
| | | Multiply by power of two | 1 | 4 |
| | | Masked load `other=0` dead value | 2 | **unknown** |
| L04 masked reduction | 22 | Wide constants | 2 | 6 |
| | | Masked load `other=0` dead value | 1 | **unknown** |
| L05 FP16 GEMM k16 | 156 | Wide constants | 23 | 69 |
| L06 FP16 GEMM k32 | 183 | Wide constants | 27 | 81 |

Across 391 total emitted words, **163 words represent identifiable, mechanised waste (42%)**.

The three mechanisms are:

- **Wide constants**: `.li` always emits four words (`movz` + three `movk`s). This is **by design** in the assembler: pseudo-instructions must be deterministically sized before expansion to avoid moving branch targets. The fix belongs in the compiler, not `.li`: a constant fitting within 16 bits should be emitted directly as a single-word `s.movz`. In GEMM, the epilogue drain loop spends four words per row for row indices 0..15, accounting for 44% of its wide constants.
- **Multiply by power of two**: `arith.muli %pid, 16` currently lowers directly to `.li` + `s.mul` (five words). Lowering constant power-of-two multiplication to a shift leaves the constant unused; this belongs in TNCA's own lowering because upstream canonicalization was measured on this module and does not perform it.
- **Masked load `other=0`**: Recorded as **unknown, not counted as savings**. It requires a mask-aware liveness fact plus a safety proof that unobserved lanes cannot fault or affect floating-point results in observed lanes. Governed by §7 R35-OPT-03; until proven, it is not a gain.

Upstream inspection confirms real optimization opportunities: TTIR generated from real Triton source contains **6 `arith.constant` ops with only 3 distinct values**, with `16 : i32`, `0.0 : f32`, and `dense<0.0> : tensor<16xf32>` each duplicated once. This provides the exact positive test case required by R35-OPT-01 — duplicate pure values in real source, rather than synthetic tests constructed to feed CSE.

Baseline bottlenecks at G0:

- The product path is TTIR → Python TNCA IR → codegen, with **no MLIR passes executed on it**.
- `opt_level` 0..2 in `python/tnca/aot.py` was marked reserved and did not alter emission.
- LinearLayout was **never invoked**; no layout computations existed to reuse.

## O0 and O1 Are Functional and Measured

`opt_level` is no longer reserved:

| Mode | Actual Behavior |
|---|---|
| B0 | Frozen baseline behavior. Unchanged. |
| O0 | Compatibility path in the new driver. Emits code **byte-for-byte identical** to B0 across all four workloads. |
| O1 | Locked upstream canonicalizer + CSE run in-place on the **module about to be imported** (so the optimized module serves as emission input); constants lower to their narrowest correct form; TNCA IR sources traverse the §4.1 bridge. |
| O2 | Layout algebra and bank-aware allocation (see below). |

Upstream passes actively transform IR: real Triton AXPY TTIR drops from 24 value operations to 20, eliminating four `arith.constant`s. The TNCA IR bridge deduplicates repeated coordinate calculations generated per descriptor in GEMM. **Equivalence is determined by upstream**: the bridge emits a pure scalar forest inside a `tt.func` returning all values, and after CSE reads the return list; two values proven equivalent return with identical SSA names. Python-side value numbering with simple upstream diffing was rejected per §4.2 as a covert replacement; upstream folds `16 + 0` to `16`, collapsing values to constants in ways structural key matching misses.

Measured on R3 v2 accepted hardware, all four pass against independent oracles:

| Case | B0 Cycles | B0 Words | O1 Cycles | O1 Words | Saved | §6.3 Threshold |
|---|---:|---:|---:|---:|---:|---:|
| L01 masked AXPY | 228 | 30 | 218 | 27 | **+10** | 12 |
| L04 masked reduction | 74 | 22 | 69 | 16 | **+5** | 8 |
| L05 FP16 GEMM k16 | 1215 | 156 | 1191 | 83 | **+24** | 60 |
| L06 FP16 GEMM k32 | 3241 | 183 | 3168 | 95 | **+73** | 162 |

**All four represent genuine gains; none individually reach `max(8 cycles, 5%)`.** L01 falls 2 cycles short; L04 falls 3 cycles short. Per §6.3, shifting thresholds to meet criteria is explicitly prohibited.

Two key observations:

- **Removing an instruction from an already stalled cycle yields zero cycle reduction.** L05 eliminated 47% of its static words but gained only 2% in cycles. In this profile, 422 of 538 stall cycles are LSU stalls, and the matrix unit stalled 544 times waiting for VRF reads. O2 must address **what the core is waiting on**, not merely what is emitted while waiting.
- **In one measurement, C05 eliminated 7 instructions but slowed down by 12 cycles**: merging coordinates extended variable live ranges. CSE without a register cost model is not free, motivating R35-ALLOC-01.

Harness fix note: threshold calculation previously used `int()` instead of `ceil()`, which rounded the L01 threshold from 12 down to 11 in its own favor. This has been corrected.

## Criterion-by-Criterion Status

Of the 27 criteria, **26 are satisfied**, with exactly 1 documented blocker.

**The current status is therefore `OPTIMIZATION_DELIVERED` — not `R35_COMPILER_ACCEPTED`.**
Per §8, sign-off is only granted when all 27 criteria pass. R35-OPT-02 is not verified. This is not due to omission: **the locked wheel does not export AxisInfo**, and deriving equivalent facts in Python is strictly prohibited by §4.2 as a covert replacement.

| Package | Criterion | Status | Evidence |
|---|---|---|---|
| R35-01 | BASE-01 | **PASS** | Six workloads, 23 launches, 47 RTL upper bound, 27 criteria, 8 mutations each with detector, frozen in `spec/r35_*.json`. |
| R35-01 | BASE-02 | **PASS** | Ten malformations rejected by name before RTL launches (resizing after the fact, dropping workloads, no longer ablating ablation targets, shortening primary comparison, changing totals to fit short matrices, lowering thresholds, removing non-regression condition, inflating budget, deleting criteria, mutations without detector). |
| R35-01 | BASE-03 | **PASS** | R3 v2 precondition verified; **O0 and B0 are byte-for-byte identical** (L01 30 words, L03 63 words, L04 22 words). |
| R35-02 | UP-01 | **PASS (locked env)** | Upstream imported from `site-packages` wheel; `linear_layout` and `passes.common` are exported symbols; layout class is `triton._C.libtriton.linear_layout.LinearLayout`. Local host fails as expected: editable install pointing to `~/src/triton` is rejected per R1-09 and §4.2. |
| R35-02 | UP-02 | **PASS** | Version mismatch, unparseable module, empty module, and malformed basis that aborts upstream each rejected cleanly; O2 and O2-no-LL are two separate compilations with distinct cache keys — ablation is a request, not a fallback. |
| R35-02 | UP-03 | **PASS** | **Changing upstream answers changes emitted code**: disabling value equivalence in the bridge reverts GEMM from 83 to 87 words; modifying the rename verdict changes L03 from 9 to 58 words. |
| R35-03 | LL-01 | **PASS** | 354 element-by-element comparisons against an independent Python enumeration that never calls LinearLayout; three seeds, twelve shapes per seed; identity mapping control group shows 12 mismatches across 16 positions (4 fixed points match). |
| R35-03 | LL-02 | **PASS** | Permutation coverage 16/16, multiplicity 1; broadcast coverage 4/4, multiplicity 2, non-invertible with inversion rejected; non-power-of-two boundaries rejected; dimension-mismatched compose rejected before reaching upstream (which would abort); layout dimensions exclude role/warp. |
| R35-03 | LL-03 | **PASS** | §3.2 three verdict classes complete: rename / real instruction / refuse; conversions emit `v.iota v.movi v.lshri v.andi v.sub v.xor v.permute`, all defined in `isa/tnca-isa-v0.1.json`; no phantom SMEM or invalid lane swaps. |
| R35-04 | OPT-01 | **PASS** | Upstream canonicalize/CSE on product path reduces real Triton TTIR from 24 value ops to 20, eliminating four frontend `arith.constant` duplicates; machine code drops from 30 to 24 words; input and optimized IR are dumpable. |
| R35-04 | OPT-02 | **NOT VERIFIED (blocked)** | Pinned wheel's `libtriton.passes.analysis` only exports `allocation` and `membar`, **lacking AxisInfo entirely**. §2 requires reusing the locked analysis; Python reimplementation is forbidden per §4.2. Planned path is a standalone tool built against Triton's own pinned LLVM (`compiler/pin/toolchain.json`), separating it from `tnca-opt` (linked against distro MLIR 20). No compiler component currently claims alignment or contiguity facts. |
| R35-04 | OPT-03 | **PASS** | Effectful operations preserve relative order across all passes (GEMM 42, reduction 3, view 3); NaN, ±Inf, −0.0, and subnormals through masked AXPY are byte-for-byte identical between B0 and O2; pass pipeline is strictly `['canonicalize', 'cse']` without reassociation or contraction. |
| R35-05 | ALLOC-01 | **PASS** | Cost model reads geometry from R3 v2 frozen contract (4 banks, 128 physical registers, 1 read per bank per cycle, low 2 bits select bank); operates on post-allocation physical registers without layout access (§3.1). Validated against hardware: t29_vrfbank on accepted core reports 47 requests, 35 accepted, 12 refused (each refusal matches one modeled cycle). Directed cases: v0/v4/v8 in bank 0 add 2 cycles; v1/v2/v3 across 3 banks add 0 cycles. Rename costs 0 moves; materialized version charges 2 `v.permute`s. |
| R35-05 | ALLOC-02 | **PASS** | GEMM shows zero resource growth from B0 to O2; neither side spills or reloads (no scratchpad exists); ownership checker cannot be bypassed. |
| R35-05 | SCHED-01 | **PASS** | **No reordering performed**; def-use, memory/effects, control, accumulation order, and async resource edges hold by construction, verified by checking op order across passes (only deletions allowed). §7 permits retaining a legal baseline when gain is unknown. |
| R35-06 | SAFE-01 | **PASS** | Five inherited test suites pass; ownership verifier runs at B0/O0/O1/O2. |
| R35-06 | SAFE-02 | **PASS** | Every pass includes positive and negative boundary cases: multiply by 16/1 becomes shift, multiply by 12/3 unchanged; constant comparison folded in 2 places, dynamic values in 0; `other` eliminated when unobserved, preserved during gather; lane-0 broadcast store narrowed, full-width store unchanged; dynamic alpha in O2 loaded from parameter buffer; renaming all identifiers produces identical 24 words. |
| R35-06 | MUT-01 | **PASS** | **All eight mutations caught by designated checkers**; all injected into isolated copies; candidate source hashes unmodified. |
| R35-07 | PERF-01 | **PASS** | All six workloads read device cycles per §6.3 C (not host wall clock, not model predictions), verified against independent oracles. |
| R35-07 | PERF-02 | **PASS** | All five conditions of §6.3 satisfied simultaneously (see below). |
| R35-07 | PERF-03 | **PASS** | 30 launches (upper bound 47), 104 seconds (bound 2400), 14,861 cycles (bound 1,500,000); no candidate search performed, well within budget. |
| R35-08 | PKG-01 | **PASS** | B0/O0/O1/O2 compile; L01 under O2 across three dynamic launches shares 1 artifact, 1 session, 1 device process; zero compiler calls after initial compilation, constant code hash. |
| R35-08 | PKG-02 | **PASS** | Five modes produce five distinct cache keys; `debug`/`dump_dir` do not alter key; compilation is deterministic; corrupted cache entries rejected; runtime-only load imports neither triton nor compiler. |
| R35-08 | OBS-01 | **PASS** | Dumps input IR, optimized IR, TNCA IR, assembly, and hex; dump and debug options do not change code; manifest records pass counts; GEMM `source_kind` is `tnca-ir`, not mislabeled as Triton, with no phantom `.ttir`. |
| R35-09 | REG-01 | **PASS** | Frozen impact scope of 13 cases all pass, 0 skipped, including legacy core-4r and system paths (included to detect collateral regressions from shared compiler changes). |
| R35-09 | EV-01 | **PASS** | Reported from measured source: mutation suite seeds complete acceptable evidence, then hides a slow/broken case behind a forged PASS. Verifier accepts honest evidence, rejects forged evidence with two explicit reasons (L06 noticeably slower, total cycles increased). |
| R35-09 | REL-01 | **PASS** | Second environment macOS/arm64 agrees with primary Linux/x86_64 across all 7 checks; **L03 across three modes is byte-for-byte and cycle-for-cycle identical** (215/61/208 cycles, 63/9/58 words). L06 cycles are excluded from cross-machine comparison due to memory model reordering spread (28 cycles on Linux); its code and correctness are verified. |

## All Five Conditions of §6.3 Are Satisfied

| Case | B0 Cycles | B0 Words | O2 Cycles | O2 Words | Saved | Gate Threshold | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| L01 masked AXPY | 228 | 30 | 206 | 24 | **+22** | 12 | Meets gate |
| L02 two-output shared intermediate | 130 | 21 | 129 | 19 | +1 | 8 | |
| L03 layout composition | 215 | 63 | **61** | 9 | **+154** | 11 | Meets gate |
| L04 masked reduction | 74 | 22 | 63 | 9 | **+11** | 8 | Meets gate |
| L05 FP16 GEMM k16 | 1215 | 156 | 1191 | 83 | +24 | 61 | |
| L06 FP16 GEMM k32 | 3235 | 183 | 3161 | 95 | +74 (±28) | 162 | |

1. **All six workloads correct** ✓ Each passes against its independent oracle.
2. **At least two workloads show measured gains meeting the gate** ✓ Three meet the gate (L01, L03, L04), two of which are not the layout micro-benchmark L03.
3. **At least one workload shows causal LL attribution** ✓ L03: O2-no-LL 208 → O2 61, **+147 cycles (70.7%)**, threshold 7. Matching evidence is copy instruction elimination: two permutations totaling 49 words disappear.
4. **No meaningful regressions** ✓ None of the six workloads slowed down; total cycles decreased (5097 → 4811).
5. **Gain sources explainable** ✓ Every gain correlates with dynamic instruction count reduction; L03 correlates directly with eliminating permutations.

**Regarding L03's 71.6% cycle reduction (154 cycles saved)**: layout `view` is a newly introduced operation in this phase. B0 is its unoptimized lowering path, not behavior from a previously released compiler. This number is measured against "materializing the view", which any lowering lacking layout algebra must do. The ablation gives the honest version of the same fact: 147 of the 154 cycles are attributable to the algebra (O2-no-LL 208 vs O2 61).

## Two Findings Regarding Measurement

- **Shared temporary directories pollute measurements**: L06 measured in isolation ran in 3241 cycles; measured immediately following L05 it ran in 3221 cycles — a 20-cycle shift, larger than several reported gains in the table. Each measurement now uses its own interpreter process and isolated temporary directory.
- **The slow-reorder memory model is non-deterministic**: The same program read five times produced 3141, 3141, 3161, 3161, 3169 (spread of 28 cycles; B0 spread 20 cycles across [3221, 3235, 3235, 3235, 3241]); under the fast model L05 read three times is consistently 1191. Thus L06 reports median across 5 repetitions with spread, and gains not exceeding its own spread are marked NOT SHOWN.

## Eight Mutations, Each Caught by Designated Checkers

| Mutation ID | What Was Mutated | Caught By |
|---|---|---|
| N01 | Wrong LinearLayout basis | R35-LL-01 |
| N02 | Tail-mask coverage check defeated | R35-LL-02 |
| N03 | Effectful operation marked pure | R35-OPT-03 |
| N04 | Async value released early | R35-SAFE-01 |
| N05 | Bank ignored in cost model | R35-ALLOC-01 |
| N06 | Opt level left out of cache key | R35-PKG-02 |
| N07 | Model cycle passed off as measured | R35-PERF-01 |
| N08 | Forged PASS hiding slow case | R35-EV-01 |

Four mutations revealed findings during development that required fixes:

- **N01**: Originally mapped all basis rows to `row`, which failed to cover the output. Upstream printed `LLVM ERROR` and aborted the process (`SIGABRT`). The checker printed nothing and caught nothing. §7 states a build failure does not count as detection, and SIGABRT is even worse. It was modified to swap two rows: still bijective and accepted by upstream, but no longer a transpose.
  This also exposed a real gap: the adapter guarded against aborts in `compose`, but not in `from_bases`. The adapter now calculates the rank of basis rows over GF(2) before calling upstream, rejecting under-ranked layouts (Gaussian elimination over bit vectors is arithmetic, not a second opinion on upstream logic).
- **N02**: After defeating the coverage report, all assertions in LL-02 still held because they only queried layouts that truly covered the output. LL-02 now tests a layout covering only half the output and asserts that it is flagged.
- **N07**: Replaced device cycles with a constant, which PERF-01 initially had no way to detect. PERF-01 now checks against a physical invariant: sequential execution of a straight-line kernel fetches every instruction at least once, so C must be at least equal to static word count. Readings below this cannot come from the hardware.
- **N04**: Designated detector is SAFE-01. Re-running the five inherited suites in disposable scratch copies tested that copy's environment: an unmodified baseline had one suite time out with all counters at zero because a lab failed to build its own bench, unrelated to code changes. SAFE-01 added `--no-inherited`, strictly for mutation testing with documented rationale; inherited suites run under REG-01 on the clean tree where they belong.

Two harness bugs were also fixed: `perf.py`'s `--only` filtered on *workload* while other checkers filtered on *case*, so `--only R35-PERF-01` matched nothing and reported 0 winners (masking the test subject). Mutation labs originally carried prebuilt benches globally, giving unbenchmarked labs a half-populated `sim/build` and a stale dependency graph.
