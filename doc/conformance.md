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

Audited against the Stage-2 + Stage-3 tree: matrix unit, TensorMap and
transfer engine on a 2x2 mesh over four NUCA slices. Re-run this audit whenever
a stage closes.

---

## The twelve frozen decisions (§"最后：我会优先冻结的十二个决定")

| # | Decision | Status | Evidence |
|---:|---|---|---|
| 1 | Data hierarchy is only RF — NUCA — DRAM; no new addressable tensor SRAM | **Held** | The chain is now complete and real: vector register → LSU beat → core NI → mesh → home NUCA slice → DRAM. Nothing in it is software-addressable except the registers at one end and DRAM at the other. The only architectural storage is `S`, `V`, `P` (ISA §1); the slice's arrays are tag- and index-addressed by physical address and no instruction can name a way, a set or a line. |
| 2 | At most four RCTXs, each a vector control context, not a scalar thread | **Held** | `tnca_core` is parameterised on `NUM_RCTX` (2 in the test configuration, up to 4). Each has its own PC, scalar file, predicate file and pipeline identity; all share one SIMD datapath and one LSU. There is no per-lane program counter and no SIMT divergence stack — vector control is masks only (ISA §2.2). |
| 3 | Regular movement goes through Tensor DMA; irregular access keeps a full LSU | **Held** | Both halves exist. The general LSU was built first, deliberately, so that the transfer engine could be added as a fast path rather than as the only path — and it remains the path for gather, scatter and arbitrary alignment. `tnca_tensor` handles regular tiles, and the two share the memory port through `tnca_mem_mux` with the transfer engine given priority, because a stalled transfer holds an RF window and a held window blocks the producer that would refill it (§2.4). |
| 4 | The matrix unit reads from RF, never cycle-by-cycle from a remote cache | **Held** | `tnca_matrix` reads its fragments from the vector file over a dedicated read port and never touches memory. The cost it avoids is now measurable rather than hypothetical: in the 2×2 run a NUCA access crosses the mesh both ways plus a slice's request buffer, tag lookup and data access, and 1,792 of 2,048 requests went to a remote slice. |
| 5 | Vector RF and accumulator RF are separate, but accumulators are reachable from the vector unit | **Held** | `tnca_acc_rf` is a separate file with its own row-wide ports, and `a.vread` / `a.vwrite` / `a.mul.row` / `a.fma.row` reach it from the vector side. §5.4's reason is the one that mattered: online softmax rescales `O` on every key block, and a round trip through the cache would have cost the design its own argument. |
| 6 | TensorMap correctness fields are strictly separated from cache/flow hints | **Held, structurally** | The TensorMap's first 128 bytes are semantics and the second 128 are hints. `tnca_tensor` parses the first half and **does not read the second at all** — there is no path by which a hint could reach an address, a value or an order, which is the strongest form the principle can take before Stage 4 gives hints something to do. The same holds for the LSU's cache-hint field. |
| 7 | Cache uses a single static home; no replication or migration in v1 | **Held** | `tnca_core_ni` computes `home = H(physical line address)` and takes no other input — no descriptor, no TensorID, no hint — so two views of one physical line cannot disagree about where it lives (§7.5, §10.2). A line exists in exactly one slice; there is no replication, no migration and consequently no inter-slice coherence protocol at all. |
| 8 | Throttle admission of new transactions, never completion or resource release | **Held** | `tnca_core`'s stall logic throttles only the front end (`stall_front`), which holds F/D/R and inserts a bubble into X. It never freezes X, the LSU or the FP sequencer. Out-of-band completions take the write port by *causing* a front-end stall, so a completion can never be starved for more than one cycle — the release path has strict priority over new work. |
| 9 | Manage MSHR entry, target, return, RF and writeback credits separately | **Partly held (Stage 4 completes it)** | MSHR entries and targets are now genuinely separate resources with separate exhaustion paths and separate counters, and running out of one does not look like running out of the other — which is the distinction §9.1 exists to make. The return path is reserved before a request is issued (the LSU's beat table), and the writeback buffer is a per-MSHR reservation taken at allocation. What is still missing is the *budget* half: `new_miss_credit` and `merge_credit` as throttled quantities with the fast and slow control loops of §9.5. That is Stage 4, and deliberately so — §21 says to build the baseline first. |
| 10 | Both "ready" and "free" events belong in the ISA and the compiler's lifetime model | **Held** | The mbarrier carries arrivals and operations as separate counts so one generation can mean more than "the data arrived" (§13.1), and `t11_matrix` uses a producer arrival plus two transfer completions in one generation. The *source-reusable* event is now expressed too: `m.mma.async` with `mod[4]` signals a second completion, on a second barrier, when it has read its fragments for the last time — earlier than the result by `FMA_LAT` cycles. `t25_mmasf` proves the two are distinct by destroying both fragments the moment source-free fires and checking the product anyway; moving the signal to the command accept turns it red. The conservative proxy §13.1 permitted for a first version is no longer needed. |
| 11 | Build the backend from TTIR and early layout analysis, not from shared-memory TTGIR lowering | **Held for the path that exists** | The frontend walks **TTIR** and nothing else — there is no TTGIR step and no shared-memory assumption anywhere in it, which is the half of this decision that can be checked before the MLIR dialect exists. `make e2e` runs a `@triton.jit` kernel through it onto the four-tile machine as a grid and checks the numbers in Python. What is still Stage 5 is the *layout analysis*: the lowering handles one element per lane and refuses anything wider (`test_unsupported_block_is_refused`), rather than choosing a layout. |
| 12 | Finish a small verifiable system before scaling the mesh and the dynamic control | **Held** | The order was exactly §21's: ISA and memory semantics on one core, then pipelining, then multiple outstanding transactions, then one slice, then the mesh, then 2×2 — each with its own test before the next was started. No adaptive policy has been added yet, which is the point: Stage 4 measures each mechanism against this baseline one at a time. |

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

**Held by construction**, and now for a concrete reason rather than an abstract
one. There is no matrix unit yet, but the cost the principle is about is now
measurable: a NUCA hit costs a mesh traversal each way plus the slice's request
buffer, tag lookup and data access. In the 2×2 run, 1,792 of 2,048 requests went
to a remote slice. Feeding a matrix array from that path per cycle was never
going to work, which is why §1.2 asks for the transfer to land in registers
first.

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

The Stage-3 cache makes this structural rather than merely stated: the hint
reaches the LSU and stops there. `tnca_core_ni` does not read it when choosing a
home, and the slice does not read it at all. There is currently no path by which
a hint could change an address, a value or an order even if it wanted to, which
is the strongest form the principle can take before Stage 4 gives hints
something to do.

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

* `CAPABILITY0.SFU` read 0 through Stage 1a, and `v.exp2.approx` trapped. It
  reads 1 now — but only because the datapath exists and has been measured
  against §8.8's bound, which is what the bit means. A build with
  `HAS_SFU = 0` still clears it, still traps, and does not instantiate the
  slice; the promise and the gates move together.
* `CAPABILITY0.MATRIX_FP8` was added with the datapath that earns it, not
  before: `m.mma.async` takes E5M2 and E4M3 fragments, `t11_matrix` runs the
  same product through all four formats against an exact reference, and
  `t14_caps` exercises the bit. One bit covers both FP8 encodings because they
  differ by one parameter in one widening step and share every other gate --
  unlike FP16 and BF16, where an implementation could plausibly build one
  exponent range and not the other.
* Major `0x21` (atomics) decoded as illegal until the home slice existed to
  execute them at. It does now, so `CAPABILITY0.ATOMICS_DEVICE` reads 1 and the
  family runs — `t22_atomic` checks every operation at both widths, and
  `t24_amo_mesh` checks the property only a multi-core machine can check, by
  having all four tiles add to one address and then spin until the total is
  *exactly* right. `SYSTEM_ATOMICS` still reads 0 and system scope still raises
  `ILLEGAL_INSTRUCTION`: there is one device, so there is nothing wider for the
  scope to mean, and executing it at device scope instead is exactly the silent
  degradation §1.6 is about.
* `s.wrspecial` traps, because v0.1 defines no writable special register.
  Letting it retire as a silent no-op would make a program that depends on it
  look correct here and fail elsewhere.

---

## Deviations

### D1 — Instruction fetch was a single-cycle asynchronous read

**Document:** §19 lists `instruction_decode.v` and a `role_frontend.v` without
prescribing the fetch timing, but a real design needs a synchronous SRAM.

**Was:** `tnca_core` drove `imem_addr` combinationally and latched the returned
word at the end of the same cycle — an asynchronous memory, which no SRAM is.

**Status: closed.** The instruction memory is synchronous: the address
presented this cycle returns its word on the next. The core carries the extra
stage (`f1_*`) and the one-deep fetch buffer (`fb_*`) that a stall makes
necessary — a stalled D cannot refuse a word that is already on its way. No
fetch is issued while the buffer is occupied, which is why one entry is enough,
and a simulation assertion fires if both ever hold a word at once.

The cost is visible and small: the same 18 core programs issue exactly the same
instruction counts and take about 8 % more cycles (`t00_scalar` 544 → 586),
which is the extra fetch latency appearing on redirects. Nothing else moved,
because nothing else depended on the timing.

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

**Still open.** The LSU handles one *instruction* at a time. Within it the
coalescing is real, and the NUCA slice behind it is fully non-blocking, so the
memory-level parallelism that Stage 3 was about exists — between cores and
within a vector access, not between two consecutive scalar loads on one core.

### D4 — The transfer engine handles rank 1 and 2, and only aligned rows

**Document:** §7.2 specifies rank 1 to 5 in hardware, with the compiler
flattening what it can.

**Built:** `tnca_tensor` executes rank 1 and 2. A tile row must also be a whole
number of 8-byte units with an 8-byte-aligned source. Both conditions are
checked before a single transaction is issued, and a violation raises
`DESCRIPTOR_FAULT`.

*This paragraph described a check that did not exist.* Only the row **size**
was tested; nothing tested where each row **started**. A 13-element FP16 row
is 26 bytes, so every row after the first began two bytes off, and the engine
returned shifted data while raising nothing. Found by compiling a GEMM with an
N tail — milestone C2, which `compiler_overview.md` §15 says is where the
tile/fragment mismatch should surface. Now checked, documented in ISA §9E.2,
and tested by `t17_descfault`.

**Why:** rank 3 to 5 is more coordinate arithmetic, not a different mechanism,
and the alignment restriction is what lets a returned 8-byte unit land in a
window register through the existing lane write enables rather than needing
byte-granular writes.

**This is the one Stage-2 item still open.** The rest of Stage 1, 2 and 3 is
closed: synchronous instruction fetch (D1), atomics and fences at the home
slice, the SFU, the five-format datapath, and `m.mma.async`'s second
completion event. Rank 3–5 needs a coordinate read the core cannot do in one
cycle — a transfer takes one coordinate per descriptor rank from consecutive
scalar registers (§9E), and the register file has four read ports — so it is a
port and a small sequencer, not a rewrite. It is named here rather than
quietly left off the list.

**Why it is a fault and not an approximation:** this is §1.6 applied to a
geometry rather than to an instruction. A transfer that quietly did something
near enough would make a wrong answer look like a right one, which is the single
failure mode a capability model exists to prevent. `TENSOR_MAX_RANK` reports the
limit so software can ask rather than discover.

**Replacement:** rank 3–5 and byte-granular window writes, together.

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

**Status: done.** The FMA is a 2-stage pipeline and the multiplier a 3-stage
one; see `synthesis.md`. The integer vector unit remains combinational.

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
   published contract, which is why the unit was absent rather than present and
   unspecified — and why it could only be turned on once measured.

   That rule is what kept Google XLS's `apfloat` out of the binary32 FMA.
   `apfloat::add(1e-45, 0.0)` returns `0.0`: the library flushes subnormals, a
   defensible choice for ML kernels and an impermissible one here. The SFU is
   the mirror image — §8.8 *requires* the flush — so the SFU is generated from
   DSLX and the FMA is not. doc/xls.md has the measurement and the reasoning.

5. **§10.5 and §13.4, forward progress.** A trap stops one role and wakes the
   rest; it never becomes a hang. `t05_fault` tests exactly that, including
   that the core still reaches idle.

6. **A multi-core test found an ISA-conformance bug.** `NUM_ROLES` is defined
   as "active RCTX count for **this launch**" (§5.1), and the core reported
   the architectural `NUM_RCTX` instead. That only ever looked right because
   every existing test launched every context. `t24_amo_mesh` synchronises
   across roles — it spins until an atomic counter reaches exactly
   `tiles * roles * N`, which is what makes a lost update a hang rather than a
   silently wrong total — and with one role launched it waited for adds that
   were never going to happen. The register now reports the launched count,
   and `t15_target` launches every role so that its target fact still holds.

   The general shape is worth naming: a register whose value is only ever
   read back by the same program that set it cannot be wrong in a detectable
   way. It took a program that *acted* on the value to find it.

7. **The compiler plan found three more.** Building
   [`compiler_overview.md`](compiler_overview.md)'s milestone C0 against the
   hardware exposed things no RTL test was asking about, because they are only
   wrong from a compiler's point of view:

   - `CAPABILITY0.TENSOR_DMA` read 0 while the transfer engine executed TXFER
     normally. ISA §11 makes a clear bit a promise of `ILLEGAL_INSTRUCTION`, so
     a compiler honouring the bitmap would have refused to emit a DMA the
     hardware performs. `t14_caps` now pins the bitmap and exercises every bit
     it claims.
   - The four resources §10.1 of the compiler plan allocates across the whole
     program — barriers, RF windows, descriptors, physical vector registers —
     were the ones the hardware did not report, so a compiler could not check
     its target description against the machine. Four read-only special
     registers were added in free encoding space, and `t15_target` now closes
     the chain from RTL parameter to special-register read.
   - A transfer whose coordinate base was `s63` silently ran at coordinate 0 on
     its second axis, because the implicit `Sb+1` wrapped to `S0`, which reads
     as zero and discards writes. Demonstrated, then made
     `ILLEGAL_INSTRUCTION` in the decoder and rejected in the assembler. This
     is precisely the implicit wide register read §13.4 of the compiler plan
     warns cannot be inferred from the explicit operand fields.

7. **Milestone C2 found two more in the transfer engine.** Compiling a GEMM
   with an N tail needs a 13-element row, and that exposed both:

   - The source alignment requirement was documented in three places and
     checked in none. See D4 above.
   - An engine fault recorded the trap but never stopped the Role Context that
     issued the command, so the program ran on past a transfer that had not
     happened, and a later `trap` overwrote the code — a `DESCRIPTOR_FAULT`
     surfaced as `TRAP_USER`. ISA §10 requires a trap to stop the raising
     context. The engine now reports which context issued and at which PC,
     because an engine fault arrives long after the issuing instruction has
     retired and whatever sits in the execute stage by then is the wrong
     answer.

8. **Milestone C3 found two more, in how a failure travels.** The two-role
   pipeline's error-propagation case: the producer's transfer faults, and the
   consumer is asleep on the barrier it was counted against.

   - The engine signalled an *ordinary* completion on that barrier while
     faulting. The barrier decremented its operation count as if the transfer
     had succeeded, so the consumer was released to read a window nothing had
     written. ISA §9A.2 gives the barrier an error bit "set by `mb.inval` or
     by a faulting command"; the engine now sets it, and the count is not
     decremented by a failed completion.
   - A waiter woken by that error then *resumed*, and went on to wait for the
     next generation of a pipeline whose producer had already died — so a
     located fault became a 400,000-cycle hang. ISA §10 lists "a waiter woken
     by `mb.inval`" under `BARRIER_FAULT`; the hardware did not implement it
     for any error source. It does now, which is also what the ISA always said
     should happen for `mb.inval` itself.

   Both are the same mistake in different places: a failure that stops one
   Role Context has to reach everything that was waiting on it, or the
   architecture's promise that a trap "must never turn into a silent hang"
   holds only for synchronous faults.

9. **Milestone C4 found a control bug that needed two Role Contexts to see.**
   A taken branch failed to squash a younger instruction of the same context
   when the front end was stalled, so the instruction after the branch
   executed anyway. In the case that exposed it, a `trap` placed immediately
   after a loop branch fired while the branch was taken.

   The cause: the stalled path killed the entry *held* in decode based on
   `kill_f` — which context the fetch stage was selecting — rather than
   `kill_d`, the context of the entry actually held. With one Role Context
   `f_sel` is always the squashed one and the two coincide, which is why every
   single-role test passed. With two, `f_sel` alternates and the kill is
   missed about half the time.

   It is worth noting how it surfaced. The symptom was a compiler-generated
   consumer spinning on a flag and trapping on a budget it had not exhausted;
   the loop was correct, the countdown was correct, and the same loop passed
   in isolation. Only instrumenting the branch condition showed the branch
   being taken and the next instruction retiring anyway.

10. **Milestone C4 also found a missing piece of the launch contract.** The
    compiler's two-role kernels began with a memory-flag handshake: role 0
    initialised the barriers and wrote a flag, and the other roles polled it.
    Under a slow memory model the compiled GEMM deadlocked — the consumer's
    polling starved the producer's transfer engine of the memory port, so the
    flag it was waiting for could never be written.

    The handshake existed because there is no race-free way to express "wait
    until role 0 has run `mb.init`" using barriers: waiting means sleeping on
    one, and §9A.6 makes `mb.init` under a sleeping waiter a `BARRIER_FAULT`.
    Any workaround has to leave the barrier mechanism, and polling memory is
    what that looks like.

    The fix is a guarantee rather than a workaround. Barriers are quiescent at
    kernel entry — which the hardware's reset and launch flush already
    provide — so no role initialises them and no role waits for another to.
    A consumer may sleep on generation 1 before the producer has executed an
    instruction. Now ISA §9A.7, and the generated code emits neither the flag
    nor `mb.init`.

11. **Milestone C4's loop cloning found a livelock in `mb.wait`.** Two
    predicates decide what a wait does: one whether to sleep, one whether to
    wake. They disagreed. The sleep test required the barrier to have been
    armed at least once; the wake test did not. A wait on generation 0 — the
    quiescent state every barrier starts in — therefore slept, woke
    immediately, re-executed the instruction and slept again.

    It was a livelock rather than a hang, which mattered for finding it: the
    machine burned ~100% of its issue slots on squashes, so the symptom was a
    program running very fast and arriving nowhere, not a stalled one.

    The `armed` term was wrong on its own terms as well. It is only ever clear
    before the first `mb.arm`, and generation 0 has trivially completed since
    nothing was ever pending. Removed, so both predicates are now
    `done_epoch >= epoch`. `t19_waitgen0` covers generation 0 before and after
    a barrier is used, and a late waiter on a generation that finished long
    ago — the case the `>=` comparison exists for, which `armed` would have
    broken as soon as a barrier were re-initialised.

12. **Milestone C4 found the two asynchronous engines sharing one completion
    port.** The core wired both into a single `opdone_en`, with
    `opdone_bar = mtx_done ? mtx_done_bar : txf_done_bar`. When the matrix
    unit and the transfer engine finished in the same cycle, the matrix won
    the mux and the transfer's completion was silently dropped — so a barrier
    waited for ever on an operation that had in fact completed.

    It needed a slow memory model to shift the timing enough for the two to
    coincide, and it presented as a clean deadlock: `issued=175` over 600,000
    cycles, with `ready[0]` stuck at `cur=2 done=1 ops=1` and the consumer
    asleep on generation 2. Instrumenting the two `done` signals showed
    exactly one coincident cycle in that run — matrix on barrier 4, transfer
    on barrier 0 — which is the dropped completion.

    The barrier now takes one port per engine, and the per-barrier decrement
    is a count rather than a flag, because nothing stops both engines naming
    the same barrier in the same cycle.
