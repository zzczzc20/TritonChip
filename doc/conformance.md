# Conformance to `architecture_overview.md`

A point-by-point audit of what is built so far against the design document.
Every row is one of:

* **Held** — implemented, and the implementation respects the principle.
* **Held by construction** — the principle constrains something that does not
  exist yet, and nothing built so far violates it.
* **Deferred** — not yet built; the stage it belongs to is named.
* **Deviation** — built differently from the document, with the reason.

Deviations are listed in full. There are three, all Stage-1 simplifications
with a named replacement, and none of them changes the ISA or the semantics the
later stages depend on.

Audited against the Stage-1a tree. Re-run this audit whenever a stage closes.

---

## The twelve frozen decisions (§"最后：我会优先冻结的十二个决定")

| # | Decision | Status | Evidence |
|---:|---|---|---|
| 1 | Data hierarchy is only RF — NUCA — DRAM; no new addressable tensor SRAM | **Held** | The only architectural storage is `S`, `V`, `P` (ISA §1). `tnca_lsu` goes straight to the memory port; there is no staging array, and no instruction can address one. The NUCA level itself is Stage 3 — until then the memory port stands in for it, which adds no storage level. |
| 2 | At most four RCTXs, each a vector control context, not a scalar thread | **Held** | `tnca_core` is parameterised on `NUM_RCTX` (2 in the test configuration, up to 4). Each has its own PC, scalar file, predicate file and pipeline identity; all share one SIMD datapath and one LSU. There is no per-lane program counter and no SIMT divergence stack — vector control is masks only (ISA §2.2). |
| 3 | Regular movement goes through Tensor DMA; irregular access keeps a full LSU | **Held (LSU half)** | The full general LSU exists now, including gather/scatter and arbitrary alignment, and it is not a fallback bolted on later — it was built first, exactly so that Tensor DMA can be added as a fast path rather than as the only path. Tensor DMA is Stage 2. |
| 4 | The matrix unit reads from RF, never cycle-by-cycle from a remote cache | **Held by construction** | No matrix unit yet (Stage 2). The structure it needs is in place: every execute-stage unit takes its operands from the register files, and the memory port is reachable only through the LSU. |
| 5 | Vector RF and accumulator RF are separate, but accumulators are reachable from the vector unit | **Deferred (Stage 2)** | Accumulator registers do not exist yet. The ISA reserves major `0x41` for accumulator access so the capability is not silently absent. |
| 6 | TensorMap correctness fields are strictly separated from cache/flow hints | **Held by construction** | No TensorMap yet. The same principle is already applied where it does bite: the LSU's cache-hint field (`mod[5:4]`, ISA §9.1) is defined as performance-only and is carried through the pipeline without reaching any address, value or ordering decision. |
| 7 | Cache uses a single static home; no replication or migration in v1 | **Deferred (Stage 3)** | |
| 8 | Throttle admission of new transactions, never completion or resource release | **Held** | `tnca_core`'s stall logic throttles only the front end (`stall_front`), which holds F/D/R and inserts a bubble into X. It never freezes X, the LSU or the FP sequencer. Out-of-band completions take the write port by *causing* a front-end stall, so a completion can never be starved for more than one cycle — the release path has strict priority over new work. |
| 9 | Manage MSHR entry, target, return, RF and writeback credits separately | **Deferred (Stage 3)** | The Stage-1 precursor is real: the LSU exposes a `req_ready`/`cpl_valid`/`cpl_ack` handshake rather than a fixed latency, and the core asks whether a unit will be free *next* cycle rather than assuming. That is the shape the credit scheme slots into. |
| 10 | Both "ready" and "free" events belong in the ISA and the compiler's lifetime model | **Deferred (Stage 2)** | mbarrier is Stage 2. Nothing built so far collapses the two: the LSU reports completion explicitly, and the scoreboard releases a destination when it is *written*, not when the instruction issues. |
| 11 | Build the backend from TTIR and early layout analysis, not from shared-memory TTGIR lowering | **Deferred (Stage 5)** | |
| 12 | Finish a small verifiable system before scaling the mesh and the dynamic control | **Held** | This is the reason the tree looks the way it does: one core, no mesh, no adaptive policy, and a regression that covers the ISA and the memory semantics before anything is scaled. |

---

## The six architecture principles (§1.1–§1.6)

### §1.1 — RF ↔ NUCA ↔ DRAM is the only data hierarchy

**Held.** The permitted exceptions are named in the document: pipeline FIFOs,
MSHRs, descriptor tables and instruction buffers, none of which are
software-addressable. The Stage-1 core adds pipeline registers and the LSU's
per-lane staging registers. The staging registers are worth stating explicitly,
because they are the closest thing here to a buffer: they hold at most one
in-flight access's bytes, they are not addressable, they are not allocated by
software, and they cannot hold data across instructions.

### §1.2 — The matrix array must not depend on remote-cache supply per cycle

**Held by construction.** See decision 4.

### §1.3 — "≤4 threads" means four vector control contexts

**Held.** See decision 2. The document's warning that role assignment is a
compiler convention rather than a hardware binding is respected: `ROLE_ID` is
readable and nothing else in the hardware treats role 0 differently from role
3. `t04_roles` runs a producer/consumer split; `t05_fault` runs two unrelated
roles, one of which faults.

### §1.4 — Semantics and hints must not be mixed; mispredicted hints may only cost performance

**Held.** ISA §9.1 defines the cache-hint field as performance-only in the same
sentence that defines it, and the decoder rejects the unallocated hint value
rather than treating it as "default". The reserved-bit rule (ISA §2.3) is the
same principle applied to the encoding itself, and `tb_decoder` enforces it with
627 vectors.

### §1.5 — The ISA should be friendly to TTIR/TTGIR without freezing IR into hardware

**Held.** The ISA's units are scalar, vector, mask, shuffle/reduce, memory —
not operators. There is no `SOFTMAX` instruction and no `FLASH_ATTENTION`
instruction, as §11.1 requires. Two places where the document's specific
guidance is followed rather than approximated:

* Triton's `tl.load` distinguishes `mask` from `other`, so ISA §2.2 defines an
  inactive lane as performing *no memory access* and leaving its destination
  lane unchanged, and leaves materialising `other` to an explicit `v.sel`. The
  alternative — forcing zero — would have silently mis-lowered any kernel with
  a non-zero `other`. `t02_mem` check 56 tests that a fully masked access
  generates no traffic at all, by pointing it at a null base.
* `tl.reduce` takes arbitrary combining functions, so only `sum/max/min/and/or/
  xor` are native and everything else (argmax, Welford, custom scans) lowers to
  compare/select/shuffle. §12.6 says exactly this.

### §1.6 — Split "broad Triton support" into three separate promises

**Held, and stated rather than implied.** ISA §11 defines a capability bit as a
promise about a *native datapath*, and a clear bit as `ILLEGAL_INSTRUCTION` —
never "works, but slowly, by some other route". This is enforced, not just
written down:

* `CAPABILITY0.SFU` reads 0 in Stage 1a and `v.exp2.approx` traps.
* Major `0x21` (atomics) is allocated and normative but decodes as illegal,
  because atomics execute at a home slice that does not exist until Stage 3.
* `s.wrspecial` traps, because v0.1 defines no writable special register.
  Letting it retire as a silent no-op would make a program that depends on it
  look correct here and fail elsewhere.

---

## Deviations

### D1 — Instruction fetch is a single-cycle asynchronous read

**Document:** §19 lists `instruction_decode.v` and a `role_frontend.v` without
prescribing the fetch timing, but a real design needs a synchronous SRAM.

**Built:** `tnca_core` drives `imem_addr` combinationally and latches the
returned word at the end of the same cycle.

**Why:** a synchronous instruction memory adds a stage between F and D and a
fetch buffer to hold the response across a stall. That is front-end plumbing
with no bearing on the ISA, the hazard logic or the memory model, which is what
Stage 1 exists to establish.

**Replacement:** Stage 1b. The change is local to F/D.

### D2 — One memory instruction in flight

**Document:** §2.3 is emphatic that four contexts are not the problem — four
outstanding loads are, and a DMA must expand into many independent cache-line
transactions.

**Built:** `tnca_lsu` handles one instruction at a time. Within that
instruction it does coalesce properly — work is driven by memory words, not by
lanes, so a 32-bit contiguous vector access costs `VLEN/2` beats on a 64-bit
port and a gather with locality collapses the same way.

**Why:** memory-level parallelism is a property of the MSHR and credit scheme
(§9.1), which belongs to the NUCA slice. Building an approximation of it against
a single-ported behavioural memory would have produced a structure that has to
be thrown away in Stage 3.

**Risk accepted:** the Stage-1 core cannot demonstrate latency hiding. No
performance claim is made from it, and none should be.

**Replacement:** Stage 3, with MSHR entries, targets and the five credit
classes of §9.1.

### D3 — The execute-stage units are combinational

**Document:** §6.2 warns against treating an arithmetic-resource ceiling as
achievable throughput, and §5.3 says to make the register-file ports real before
deciding the compute.

**Built:** every non-memory unit produces its result in the cycle it is issued.

**Why:** single-cycle execute made the hazard, squash and completion logic
small enough to get right, and four of the bugs the regression found were in
exactly that logic.

**Measured, not assumed:** `make synth` reports 207 gate levels for the FMA and
112 for the scalar ALU, against a ~50-level budget at 1 GHz. This is recorded in
[`synthesis.md`](synthesis.md) as a finding, and the honest reading is that the
Stage-1 core would clock at roughly a quarter of the target. The document's own
advice is the reason this is reported rather than quietly left: §20.1 says a
roofline is a lower bound, not an execution-time formula, and a clock the design
cannot reach is the least defensible input to any such bound.

**Replacement:** Stage 1b pipelines the FP unit and the scalar multiplier. The
core is already shaped for it — a per-unit latency and an out-of-band completion
path exist — so the addition is a writeback reservation shift register.

---

## Places where the document changed the implementation

Worth recording, because these are cases where following the document cost work
that a simpler reading would have skipped:

1. **§6.2, "don't treat the MAC ceiling as throughput", applied to the
   multipliers and then to the whole vector unit.** The first vector ALU
   synthesised to **205,205 gates** — larger than the entire 16-lane FP array —
   because it built an unsigned *and* a signed multiplier per lane for `mulh`
   (an operation on no inner loop), five separate crossbars for five shuffle
   patterns that differ only in a source index, and six comparator trees where
   one subtractor's borrow bit answers all of them. Rewriting those three
   structures brought it to **145,334 gates (−29 %) with identical behaviour**,
   verified by the same 266 vector checks before and after; the scalar ALU
   dropped 23 % the same way. `t06_alu` covers the signed-high-product
   derivation with 186 randomised cases and was confirmed to fail when the
   correction is removed. See [`synthesis.md`](synthesis.md).

   This is the clearest example of the document changing the design rather than
   describing it: none of these were visible from the RTL, and all three would
   have shipped without the measurement step §20 asks for.

2. **§8.7 and §12.6, reduction semantics.** `vf.reduce.add` has an
   architecturally fixed balanced-tree order, so the result is bit-reproducible
   across implementations instead of being whatever the hardware found
   convenient. The RTL folds through the existing FMA array one level per
   cycle rather than instantiating 15 dedicated adders.

3. **§12.2, floating-point min/max.** Three explicit NaN modes
   (`.ieee`/`.prop`/`.fast`) with signed-zero behaviour specified per mode,
   because the document names inconsistent lowering as a known hazard. All three
   are tested.

4. **§12.4 and §12.7, no silent approximation.** FTZ is an explicit encoded
   mode, never an implementation shortcut; the SFU's accuracy bounds are a
   published contract, which is why the unit is absent rather than present and
   unspecified.

5. **§10.5 and §13.4, forward progress.** A trap stops one role and wakes the
   rest; it never becomes a hang. `t05_fault` tests exactly that, including
   that the core still reaches idle.
