# Hybrid LSU and opt-in core integration

The compact vector planner is now connected to the original narrow two-unit LSU in tnca_lsu_hybrid. Core parameter HYBRID_LSU=1 selects it with a 512-bit memory port and four narrow beat credits per unit. HYBRID_LSU defaults to zero; the previous one/two/general-bank branches remain available. The tile/mesh/cache and board wrappers do not enable this mode yet. This is progress toward the end-to-end candidate, not a new image or acceptance result.

## Ordering and receiving capacity

Eligible within-line contiguous vectors use the shared planner and eight response slots. All other accesses, including gather/straddles, scalar MMIO and atomics, use the original 64-bit pair. A mode switch drains the other engine and its pending completion. Atomics start only at complete quiescence and block all younger admissions until their completion retires at the core. This preserves ordering without speculative address-disambiguation machinery.

The outer four-bit LSU tag partitions into a path bit and three engine bits. Fast commands use eight identities; the narrow pair uses two units with four beat identities each. No identity expands silently beyond the existing core memory-mux budget. A real single-entry completion buffer accepts an inner completion before releasing its slot and holds data/metadata stable under RF backpressure. Fence-ready includes this buffer. Tagged out-of-order completions remain supported. The auxiliary scalar completion field for vector loads matches the original LSU's lane-zero sign/zero extension, even though vector destinations consume the vector field.

The word/line adapter places each narrow request's data and byte enables at its original eight-byte position in a 512-bit word, preserves its original address and scalar compare operand, and remembers the position per outstanding tag for response extraction. Keeping the original low address bits is necessary for future scalar MMIO decoding. The core's tensor engine stays at its original 64-bit interface in hybrid mode and uses another adapter before the unchanged owner-tag memory mux; its transfer semantics are not rewritten as wide transfers.

Counters distinguish fast admitted/buffered instructions, slow admissions, architectural retirement, fast retirement, request/slot stalls, peak fast occupancy, cross-mode stalls and actual RF writeback stalls. An inner buffered completion is explicitly distinct from core retirement. These counts observe command-valid at this module; earlier pipeline waits require the core stall counters too. The new signals still need exposure through the complete board counter interface.

## Mixed functional matrix

`tools/r41/lsu_hybrid_tests.py` passes twelve cases, each with 600 commands: three seeds, ready/stalled memory and mixed or vector-burst programs. The reference is the existing narrow pair with its original eight beat credits per unit. Each engine has independent randomized memory latency, response reordering, request stalls and completion backpressure. Every command's complete scalar/vector result, metadata and lane mask is compared, as is the full final memory. Request counts are deliberately allowed to differ because the fast path coalesces a vector into one transaction.

The memory agents execute the original shared atomic functions. Cases include naturally aligned 32-bit atomics in both halves of an eight-byte word, 64-bit atomics, and known store/CAS-success/read/CAS-failure/read chains. Vector bursts actually occupy all eight fast slots. Request and completion stability, tag ownership, drained transactions and admission/retirement counters are checked. Mixed narrow-heavy traffic can be slower than the reference; vector-burst cases are faster. These are RTL fixtures, not operator/board speedups.

`tb_lsu_hybrid_reset` passes full-fast-bank abort/recovery, an atomic at the last word of a line, an attempted younger vector while atomic completion is held, scalar MMIO word placement, and reset with a buffered completion. The MMIO test is an opaque data/address transport check; it does not implement the actual TMU.

The first miter timeout was an inherited driver scheduling issue: readiness now depends on the new command's eligibility and the current memory-ready input. The driver changed both and sampled readiness in the same delta cycle. The corrected driver lets these combinational paths settle before offering an instruction. The failure log remains in initial.log; no DUT command-dropping workaround was introduced.

## Native core execution

`tools/r41/hybrid_core_tests.py` builds the actual core with HYBRID_LSU=1, its tensor adapter and four parallel SFU slices, then runs existing assembler programs with a tagged, delayed, backpressured and reordered 512-bit memory fixture. The fixture preserves the original 64-bit instruction-image format and synchronous instruction fetch. It is not a NUCA, AXI or HBM timing model.

All four programs pass their existing in-program numerical assertions:

| Program | Fast / narrow LSU instructions | Retired | Meaning |
| --- | ---: | ---: | --- |
| t02_mem | 5 / 28 | 33 | Sizes, sign/zero extension, masking and gather/scatter |
| t22_atomic | 0 / 176 | 176 | All existing atomic operations and untouched bytes |
| t12_tensor | 0 / 0 | 0 | Original tensor engine executes through the wide-word adapter |
| t37_lsu_straddle | 59 / 614 | 673 | Unaligned vector accesses and partial stores across boundaries |

Core counters prove that actual instructions enter both paths; total admissions equal architectural retirement. These single-core programs exercise the bench's existing default frontend/VRF configuration, not the complete Stage-2 image configuration or the original 113-point board suite. Their fast peak is one; high-concurrency receiving capacity is established separately by the LSU burst matrix, and still needs a compiled-operator/core test.

The first native run timed out because the new fixture's pool-valid vector was initialized to zero at declaration, leaving its always-@* free-slot calculation without a reset transition in Icarus. Ready stayed unknown and no memory request transferred. The fixture now initializes through reset; the DUT was not changed to evade the failure. The initial timeout and diagnostic trace are retained. The corrected run, source/binary hashes, native inputs and logs are in native_core_results.json and native_core_raw.tgz (compiled simulation binary excluded). The minimal-exit diagnostic also passed.

## Physical evidence and remaining gates

Combined hybrid synthesis reports **30,658 LUTs and 15,253 FFs**, including the compact planner, original fallback pair, adapters, completion buffer and counters. Completed synthesis reports are under screen_v1/out/. The free 10 ns SLR1 route completed successfully on acf3019 under ~/r41_hbmbw_hybrid_lsu_v1, using the shared slot wrapper and four threads. Routed setup WNS is +1.271 ns, hold WHS +0.011 ns and TNS/THS zero. Routed area is 30,429 LUTs and 15,247 FFs. Complete reports and the exit-zero log are in screen_v1/. OOC boundary ports are not physically pinned; this does not qualify combined-core or image timing.

Next enable the wide NI/mesh/cache path, implement wide-home atomic/MMIO positioning, expose the new controls/counters through the hierarchy, and test the real compiled standard torch points with this exact implementation. The prior generic wide-unit model speedups are hypotheses, not predictions validated for this new dispatcher and completion buffer. The 100 GB/s memory-system target and several-times board operator gate remain unproven. No full image run or card access occurred.

## Coupled tile/mesh/cache/AXI integration

HYBRID_LSU and SFU_LANES now propagate through the board kernel, FPGA wrappers, persistent shell, top and tile. Hybrid mode uses a 512-bit core and cache word with a 720-bit NoC payload; external memory lines remain 64 bytes. The coherent host mux positions its 64-bit data and byte enables within the wide word and remembers the response position. The default parameters retain the 64-bit path.

The first one-core masked-vector grid produced correct memory results, but a subsequently added admission counter assertion exposed zero compact admissions. The masked stores were redundant with earlier scalar stores, so result equality alone did not detect dropped requests. Diagnosis showed eligible vector commands at aligned addresses. The actual integration bug was the core's R-to-X capacity reservation: R checks readiness before X supplies the command, whereas hybrid readiness selected a path using the current X command. A command could reach X while its selected engine was still draining.

The hybrid core now reserves a command-independent landing register in R, captures every operand in X and holds the complete request until the selected engine can accept it. Fences and atomic idle checks include this register. An overflow assertion protects the reservation. This preserves finite fast-engine concurrency instead of requiring the entire hybrid LSU to drain before every command. The coupled matrix is being rerun with actual fast/slow admission and retirement checks. The raw initial functional pass and subsequent coverage failures remain in coupled/.

The legacy one-core grid regression passes. A separate legacy masked-vector run with 300-cycle AXI B delay completed its functional checks but failed the pre-existing requirement that the shell finish before outlet drain; that delay did not guarantee such an overlap for this program. It is retained as a failed test and is not an RTL correctness or throughput qualification. The normal-delay hybrid matrix is the next functional gate; a dedicated drain stimulus is still needed for the delayed-B coverage requirement.

The initial wide-kernel diagnostic synthesis was explicitly stopped (Vivado PID 9405) after the reservation bug was found. Its source snapshot lacks the landing-register fix and cannot qualify the candidate. No full link or card operation was performed.

With the landing register, the one-core coupled masked-vector test passes through the real NI/mesh/home/outlet path: 58 fast admissions, 1,310 narrow admissions and 1,368 retirements at the end of the write grid (including the timeout probe). The second verification grid, all 455 host reads, physical-error propagation and refusal checks pass. The three-core/two-outlet and three-core/three-outlet cases remain running under GB10 PID 337563; landing_matrix_partial.log records the completed one-core case only. These are functional results, not operator-speed measurements. The corrected free kernel synthesis is running on x86 under ~/r41_hbmbw_wide_kernel_v1, screen_landing.log and out_landing (launcher PID 10034), with four threads and the required environment. Native hybrid ISA regression is also being rerun against this core revision.

## Full-vector data-path regression

The separate r41_hybrid_full_grid.s program retains the scalar/masked checks and adds full 16-lane vector store, load and copy to a second region. Each of 17 programs owns distinct 64-byte lines. The host checks all eight 64-bit words in both regions against the exact repeated 32-bit value, covering all 16 lanes (272 word checks per configuration). The verification grid also checks original scalar results. All three configurations pass: one core/one outlet, three cores/two outlets and three cores/three outlets. Mesh, dispatch and home links have two stages in hybrid mode; home data is pipelined. The small cache geometry is four sets/two ways, so this is not the complete image profile. Actual fast and narrow admissions are nonzero and their sum matches retirements before verification. Raw log: coupled/full_vector_matrix.log.

The corrected native ISA regression also passes all four programs; coupled/native_landing/ retains source hashes, counters and logs. These tests establish functionality and ordering only. They do not establish sustained throughput, complete instruction coverage, image timing or standard torch speedup. The diagnostic three-core/three-256-bit-outlet configuration has a 9.6 GB/s outlet upper bound per direction, and cannot establish the 100 GB/s end-to-end target; end_to_end.md records this architecture gate explicitly.

## Standard torch coupled-RTL entry point

The 455-program masked matrix has now completed all three configurations successfully; coupled/landing_matrix.log is the final log. It includes timeout/drain checks, fast/narrow retirement accounting, verification grids, coherent host reads and physical-error refusal.

The MMIO RTL stand-in now accepts explicit hybrid_lsu and sfu_lanes options, forwarding them into both regular and board register benches. Defaults remain the existing 64-bit/scalar-SFU geometry. tools/r41/hybrid_torch_rtl.py runs ordinary torch.compile operations for add, mul, silu, gelu and clone/copy on TNCA and compares float32 results with the original CPU operation using unchanged torch.testing.assert_close defaults. It requires actual TNCA launches. This is a functional test, with no board speedup claim.

The first launch failed during Triton backend discovery because the test imported the TNCA driver before importing Triton; the script now imports Triton first, matching the existing native harness. The failure is retained in coupled/torch_import_failure.log. The corrected 33-element test is live on GB10 PID 349929, log /tmp/r41-wide-coupled/torch_hybrid_retry.log, with at most two compilation threads and a 16 GiB virtual-memory limit. It has elaborated the actual multicore MMIO RTL and entered vvp. Numerical results remain pending; do not count any of the five operators as passed yet.

The corrected 33-element standard torch test has completed: all five operators pass the original float32 assert_close checks on the actual wide multicore RTL. The add region contains 27 compiler tuning launches, so no speedup is calculated from it. The retained torch_hybrid_pass.log is the authoritative functional result. A matched 64-bit baseline and a one-element diagnostic are still live on GB10 (PIDs 355234 and 357316). Optional trace observation now records MMIO doorbells and hybrid engine counts in the simulator log, with a nonblocking observation-cycle counter to avoid monitor scheduling races. A new 33-element observed run (torch_observed33.log) will also retain per-launch RTL cycle values; those results remain pending.

## Observed operator admissions

The observed 33-element run completed all five operators. Its 31 MMIO grids include the first add's compiler tuning. The boundary observer reports 36 compact admissions and 504 narrow admissions, with all 540 retired; hybrid_trace_summary.py independently checks the per-grid counter deltas and retains them in coupled/torch_observed33_summary.json. The observer does not label grids with operator identities. The final five grids admit compact requests, establishing actual use of the wide path, while earlier tuning grids use only narrow requests. This is not a bandwidth result. Raw numerical/cycle output and boundary observations are retained as torch_observed33.log and torch_observed33_boundaries.log. The matched legacy 33-element functional run also passes all five operators (torch_legacy_pass.log).

The test now supports an explicit warmup before its collected launch interval, keeping compiler tuning out of the comparison. A matched 257-element pair is live on GB10, hybrid PID 574840 and legacy PID 574841, logs /tmp/r41-wide-coupled/torch_warm257_{hybrid,legacy}.log. Both use the same program/compiler settings and small cache fixture; the candidate enables the hybrid LSU and four SFU slices. These are synthetic-memory RTL comparisons, not the author-required board speedups. Larger sizes, original dtypes and complete ArchBench coverage remain required.

## Warmed coupled-RTL comparison: first 257-element results

The matched float32 baseline run has completed all five operators. The candidate has so far completed add, mul, silu and gelu; copy remains pending. Each recorded interval contains exactly one launch after a numerically checked warmup, with zero tensor upload/download bytes. Preliminary device-cycle results are:

| Operator | Legacy RTL cycles | Hybrid RTL cycles | RTL cycle ratio |
| --- | ---: | ---: | ---: |
| add | 1,805 | 1,209 | 1.493x |
| mul | 1,855 | 1,102 | 1.683x |
| silu | 1,697 | 1,194 | 1.421x |
| gelu | 3,244 | 3,015 | 1.076x |
| copy | 1,223 | pending | pending |

These are small-cache synthetic-memory RTL results, not board speedups or the 100 MHz Stage-2 board baseline. They do not meet the several-times operator gate. The particularly small gelu improvement is retained without redefining acceptance. The complete baseline and candidate prefix are in coupled/torch_warm257_legacy.log and torch_warm257_hybrid_partial.log. compare_torch_rtl.py requires complete five-operator runs, one positive measured launch per operator, unchanged shape/dtype, successful numerics, warmup and zero tensor staging. Its rejection of this incomplete candidate has been verified; no complete comparison JSON is claimed yet.

Optional CORE COUNTERS observation now records active cycles, issued instructions, exclusive dependency/LSU/FPU/VRF stall reasons, frontend cycles and home hit/miss/full counters at grid completion. A live trace has elaborated and emitted those counters; core_stage33_partial.log is a partial diagnostic snapshot. Cache deltas between completed grids also include intervening coherent host transactions, so those deltas must not be attributed solely to operator traffic. The warmed 33-element stage run remains active (GB10 PID 580085).

A candidate experiment allows a maximum compiler block of 128 instead of 16, to test whether amortizing per-program work helps the real compact engine. It is live on GB10 PID 582347, log torch_warm257_block128.log. The test now records block limits and selected grids; it also accepts float16 for subsequent coverage, but no float16 run has passed this new entry point yet. These options do not alter arithmetic or tolerance. The integrated outlet/whole-image/board gates remain open.

The candidate 257-element run has now completed all five operators. Copy takes 836 cycles versus the legacy 1,223 (1.463x). The strict comparison succeeds on the complete pair; coupled/torch_warm257_comparison.json retains exact ratios and input hashes. The earlier partial log and refusal remain historical evidence. All five ratios are below 1.7x, so this diagnostic candidate does not satisfy the several-times gate. No board speedup has been measured. The stage and block-128 experiments remain live.

The warmed stage-counter 33-element run has completed all five numerical checks. Complete operator output and cumulative per-core observations are retained as coupled/torch_stage33.log and torch_stage33_counters.log. Counts include the tuning/warmup grids; do not treat cumulative counts as a single measured operator interval. The block-128 experiment is still running: its initial add/mul/silu results are 1,342/1,329/1,892 cycles, slower than block16 at this small shape. The selected grid is three programs, so increasing block size must not be promoted on a model prediction alone. Its partial log is retained.

A paired 16,384-element float16 run has been launched with unchanged numerical comparisons and a checked warmup: legacy block16 and hybrid maximum block128. These larger-shape results are pending and remain RTL evidence only. They test a size/dtype used by the memory-bound suite, without claiming the complete ArchBench or board gate.

## FP16 memory-sized standard torch RTL comparison

At 16,384 float16 elements, all five warmed, resident standard torch operators pass the original numerical checks. Legacy block16 versus hybrid/four-slice-SFU maximum block128 gives the following RTL device-cycle ratios: add 3.530x, mul 3.623x, silu 2.509x, gelu 1.898x, copy 4.150x. Each measured interval is one launch with zero tensor staging. Raw logs and strict input-hashed comparison are coupled/torch_fp16_16384_{legacy16,hybrid128,comparison}. The selected grids are 1,024 versus 128 programs; these ratios include the compiler-block difference, not just the memory-path hardware. They use the small-cache synthetic-memory RTL stand-in, not the 100 MHz board image; the required board ratios and 100 GB/s system target remain unmeasured. The hybrid run predates enabling HIT_CHAIN in the tile. Gelu remains below a several-times result even in this fixture. *[Audit F13: the legacy control also runs one SFU slice against four. The matched-block ratios are in `coupled/torch_fp16_16384_comparison_block{16,128}.json` (add 1.763x / 2.432x, gelu 1.060x / 1.185x). Use those, not the headline above.]*

To isolate the block effect, new warmed 16,384-element float16 runs use the same block on both sides: legacy maximum block128 (GB10 PID 799986) and hybrid-with-chain block16 (PID 799987). They are live under /tmp/r41-wide-coupled/torch_fp16_16384_{legacy128,hybrid16_chain}.log. These further tests do not replace board measurements.
