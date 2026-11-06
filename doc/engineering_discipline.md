# Engineering discipline: evidence integrity, the standard path, and no special-casing

**Status: binding.** This document applies to every contributor to this repository, human or AI agent (Claude, Codex, Kimi, Gemini/agy or any other), and to every task, branch and report. It takes precedence over any task brief, handoff, plan or goal statement. A task brief cannot grant an exception to it; only the project author can, in writing, for a named case.

It exists because work in this project has been presented as progress when it was not:
- measurements were typed into code as literals and reported as "measured";
- acceptance items were judged "passed" from mock devices;
- decisions were attributed to "the lead" that the lead never made;
- performance work was steered toward fixed-function side paths that serve one stage target instead of the architecture.

The rules below make that impossible to do without visibly breaking a rule.

## 1. Definitions

- **Standard path.** A public PyTorch program (eager or `torch.compile`) on `device="tnca"`, lowered by the project's normal stack: the Inductor/Triton backend and the TNCA compiler, the runtime and the device. It uses default settings, and every operation is executed by the architecture's general mechanisms: the ISA, the tensor transfer engine, the cores, the NUCA hierarchy and DRAM.
- **Special-casing.** Any mechanism that makes a measured case faster or "passing" without making the general class of programs it represents faster:
  - recognising benchmark names, shapes, sizes, dtypes, layouts or graphs;
  - code paths or hardware that serve only the measured operators;
  - hidden or non-default knobs needed for the result;
  - hand-written kernels outside the standard path;
  - side APIs that a normal PyTorch user would not reach.
- **Constructive exception.** Adding a new mechanism (a unit, an engine, a CU, a runtime path or a compiler pass) whose purpose is to route around the existing architecture or infrastructure rather than to improve it, so that a target is met by an exception instead of by the system. Examples: a fixed-function copy or arithmetic engine reachable only for a few operators; a second execution path that bypasses the cores or the compiler; a runtime branch that only triggers for benchmark tensors.
- **Test poisoning.** Shaping the test so that it says what is wanted:
  - choosing, dropping, reordering or resizing cases after seeing results;
  - changing baselines or references, loosening tolerances after looking at the device output;
  - excluding unfavourable operators or sizes;
  - measuring a boundary that hides the cost;
  - comparing against a weakened control;
  - declaring an operator class "out of scope" to escape a regression.
- **Fabrication.** Reporting anything as observed that was not observed:
  - a number not produced by a recorded run;
  - an estimate or model output labelled a measurement;
  - a mock or simulated device standing in for the board in an acceptance claim;
  - a quote, approval or decision attributed to the author, the lead or a reviewer that they did not give.

## 2. Rules

### 2.1 Evidence integrity
1. Every number in a report, status file, commit message or acceptance index comes from a recorded run. The raw output is kept in the repository, or recorded with its sha256 and host path. The command, the source commit, the image hash and the clock are recorded with it.
2. Measurements are never written as literals in code. A script that reports a result computes it from the run it performs or from the raw files it reads.
3. Model (SystemC/TLM), RTL-simulation and board results are labelled by domain everywhere they appear. A model prediction or an RTL-simulation result never closes an item that requires the board.
4. Mock, fake or stub devices are allowed only to test harness logic, and only in unit tests named as such. They never produce a result that is reported as performance, correctness or acceptance.
5. Failures are preserved. A failed run keeps its raw output, and a later success is recorded beside it, never instead of it. A failed item stays failed until its cause is fixed and it is re-measured.
6. References and tolerances are fixed before the device result is examined, and derived independently (CPU references, float64, reference rounding floors). They are never adjusted to make a device result pass.
7. No one quotes or paraphrases the author, the lead or a reviewer without a source. A source is the message, issue or file in which they said it. "The lead decided" without a source is fabrication.

### 2.2 The standard path is the only measure of performance
1. Performance targets are measured through the standard path, on standard PyTorch operators, with default knobs. A result that needs a non-default environment variable, configuration flag or API is not a result for the target. If a non-default setting is genuinely better in general, it becomes the default for everyone, after the non-regression gate (§2.4).
2. No code, in the compiler, the runtime, the harness or the RTL, may recognise a benchmark, a test case or a point by name, shape, size, dtype or graph structure in order to treat it differently. Generic legality conditions (alignment, contiguity, dtype support, sizes that fit a resource) are allowed when they are the same for every program and stated in the architecture documents.
3. Improvements must come from the general mechanisms, so that a broad class of programs benefits, not only the measured operators:
   - the ISA;
   - the tensor transfer engine and its descriptors;
   - the cores' load/store units, register files and functional units;
   - the NoC and the NUCA hierarchy;
   - the memory outlets and DRAM;
   - the compiler's lowering, scheduling and tiling.
4. Constructive exceptions are forbidden. No new unit, engine or execution path may be added whose effect is to bypass the cores, the compiler or the NUCA hierarchy for a selected set of operators, unless the author approves that unit in writing as part of the architecture. Fixed-function accelerators reachable only for a few operators are constructive exceptions, whatever they are called.
5. Architectural principles stay intact (`doc/conformance.md`, `doc/architecture_overview.md`, the ISA): the data hierarchy is register file, NUCA, DRAM, with no new software-addressable tensor SRAM; instruction semantics and numerics do not change; and the compiler, not hand-written kernels, produces device code.

### 2.3 Benchmark integrity
1. A performance target is evaluated on a benchmark set that is **registered before the work starts**: operators, dtypes, sizes, layouts (including non-contiguous and broadcast cases), baselines and the measurement boundary. Results are reported for every registered case, favourable or not.
2. The set covers the class of programs the target claims, not only the cases expected to improve. For memory-bound work, that is at least:
   - copy and clone;
   - element-wise binary operators (`add`, `sub`, `mul`, `div`);
   - unary activations (`relu`, `silu`, `gelu`, `exp`, `sigmoid`);
   - casts between fp32, fp16 and bf16;
   - reductions (`sum`, `mean`, `max`) and normalisations (`layernorm`, `rmsnorm`, `softmax`);
   - small and large sizes, contiguous, strided and broadcast cases.
3. The baseline is the latest delivered image and software, measured with the same harness and the same boundary. Baselines are not re-measured selectively, and the comparison never uses a weakened control.
4. GEMM and other compute-bound operators are excluded from a bandwidth target only because they cannot show a bandwidth effect. They stay in the non-regression set. Excluding a class from a target is never a licence to degrade it, or to tune the tested cases at its expense.
5. The ArchBench suite (`archbench/`) and its efficiency metric (eta against the platform's spec bound) are the shared yardstick. Changing a bound, a peak or a demand count needs a written justification from the hardware facts, made before results are compared.

### 2.4 Non-regression
1. Every change keeps the 113 original points' numerics and default code, unless the change's purpose is a general improvement of that default; that is checked with `tools/performance_harness/default_change_audit.py`. `make compiler` and the RTL regressions pass.
2. A performance change must not slow down any registered point beyond its measured noise without an explicit, reported trade-off approved by the author.

### 2.5 Resources: subtract as well as add
1. Before adding hardware, measure where the existing resources go: per-module LUTs, FFs, BRAM, DSP and CLB packing per SLR from the latest routed image. Show what the addition costs against what could be removed.
2. Remove what is redundant, oversized or unused: duplicated logic, debug or trace structures the delivered image does not need, over-provisioned queues, tables and windows, unused parameter combinations. Each removal needs a measurement of its saving and a regression proving that no semantics change.
3. Removing an architectural capability (an instruction, a numeric format, a precision, an SFU function, a semantic range) is not a resource optimisation. It needs the author's approval (the R41 contract already forbids trading such capabilities for timing).

### 2.6 Process
1. Only the author approves the use of shared resources outside written pre-authorisations: the shared U280 card, new images, full implementation runs. No agent claims or implies such an approval.
2. An acceptance or performance claim carries a reproducible command and its evidence path. A reviewer must be able to rerun it.
3. Work stays inside its stated scope and branch rules. Hidden scope changes, "temporary" bypasses and unreported deviations are violations.
4. When a target cannot be met honestly, report that, with the measured bottleneck. An honest negative result is acceptable; a manufactured positive one is not.

## 3. Enforcement
- **Reverts:** any commit that breaks §2.1 or §2.2 is reverted as soon as it is found. The revert and its reason are recorded in the relevant status file. The rest of that contributor's output on the same task is re-audited before anything built on it is accepted.
- **Rejected results:** results produced under special-casing, constructive exceptions or test poisoning are not counted for any target, even if correct.
- **Briefs:** every task brief and handoff must reference this document. A contributor who receives a brief that conflicts with it follows this document and reports the conflict.

## 4. Checklist for every reported result
Before a result is reported, the contributor confirms each line, and a reviewer can check each one:
- [ ] The number comes from a run whose raw output, command, commit, image hash and clock are recorded.
- [ ] Its domain (board, RTL simulation, SystemC/TLM model, analytic estimate) is stated next to it.
- [ ] It was produced through the standard path with default settings, or the deviation is named and the result is not counted for a target.
- [ ] No code recognises the benchmark, its shapes or its names.
- [ ] No mechanism used bypasses the cores, the compiler or the NUCA hierarchy for selected operators.
- [ ] The benchmark set and baseline were registered before the work, and every case is reported.
- [ ] References and tolerances were fixed before looking at the result.
- [ ] Failures are preserved alongside successes.
- [ ] Every quote or decision attributed to someone has a source.
- [ ] The non-regression gate passes.
