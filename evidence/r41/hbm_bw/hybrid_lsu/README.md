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
