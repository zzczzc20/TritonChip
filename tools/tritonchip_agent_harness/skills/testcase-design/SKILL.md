---
name: testcase-design
description: Design test cases of your own for a TNCA compiler or machine change and measure them the way the hidden evaluation does (its correctness protocol, its device-cycle timing, its roofline bound). Use when writing, choosing or running test cases, when judging whether a result is correct, or when deciding what a change must be measured on.
---

# Designing and measuring test cases of your own

You are measured on a benchmark you are not shown. What you are shown is the
class of workloads and the rules by which a program is judged, timed and
scored. This skill (at `/opt/tnca-skills/testcase-design`) is how to turn those rules into test cases of your own,
and how to measure a case exactly the way the evaluation would, so that what
you see in your sandbox predicts how the hidden benchmark will see your
change. It names no benchmark program; it gives you the method the benchmark
is built with.

```text
scripts/protocol.py   the evaluation's correctness protocol: reference, criterion, tolerances
scripts/bound.py      the evaluation's roofline bound: demand, spec_peak, L, eta
scripts/tnca_case.py  measures a case file the evaluation's way (verdict, cycles, fallbacks, eta)
examples/             the matrix-multiply class expanded into its dimensions, as runnable cases
```

## 1. What a case is, and how it is judged

A case is a PyTorch module, its inputs (CPU tensors), the dtype it computes
in, and a tolerance. The evaluation runs every program through the same
chain you are given: `torch.compile` with TNCA as the backend, compiled once
and run once. Nothing is hand-written below PyTorch.

**Reference.** Eager PyTorch in binary32 on the CPU, from the same module and
inputs with every low-precision weight and input upcast exactly, TF32 off.
The tolerance therefore measures the device's arithmetic, not the reference's.

**Criterion.** `max |got − ref| / max(max |ref|, 1) ≤ tol` over every
output. NaN positions must agree. Integer outputs (an argmax, gathered
indices) must be exact. The tolerance is set by the dtype the case computes
in: 2·10⁻⁴ for FP32, 2·10⁻² for FP16, 6·10⁻² for BF16, and 5·10⁻² for a whole
model in FP16.

**A host fallback fails the case** even when its numbers are right: any
computation Inductor leaves on the host (an extern kernel, an ATen call, a
C++ kernel) means the chain was skipped. `tnca.inductor.host_fallbacks(code)`
finds them in the generated wrapper; `scripts/tnca_case.py` reports them.

**Each case is judged on its own.** A failing case is a failed case; it does
not spoil the others. The evaluation's status names are `pass`,
`numeric-error`, `compile-failure`, `runtime-fault`, `capacity-exceeded`,
`timeout` and `unsupported` (host fallback included).

**Do not use `torch.testing.assert_close` as the criterion.** It is per
element and, for FP16, about twenty times stricter than the protocol, so it
rejects changes the evaluation accepts, such as a different accumulation
order; and it says nothing about NaN positions.

## 2. How a case is timed

Device time is the sum, over every launch the program makes, of that
launch's device cycles: from the launch's start to its end, as the device
counts them. How the device runs a launch inside, on one core or across
several, is the machine's business; the quantity is the same on every
machine the evaluation has. Host transfers are not device time: every launch
uploads and downloads its arguments, and that is reported separately.

Two things to know about the number:

- `scripts/tnca_case.py` reports the cycles the device reports to the driver. The
  score counts at the device boundary, with the platform's own observer,
  which is a few cycles more per launch. Compare cases and versions by their
  relative cycles; the absolute number is not a score.
- You measure the device your environment points at. The task text says
  which machine that is, how it is built, and which settings select it; the
  script takes those settings from your shell exactly as your own Python
  would. After any change to the RTL, to a profile, to a device description
  or to a compiler target, build again as the task text says, or you are
  timing the old machine.

## 3. Where you stand: the roofline bound

Every program has a lower bound L in cycles that no program for this machine
can beat: `L = max over resources of demand / capacity`. Efficiency is
`η = L / T`, where T is the measured cycles. η tells you how far a case is
from the machine's limit and **which resource binds**, which is the single
most useful thing to know before deciding what to change.

**Demand** is the semantic work of one call, independent of any program:
dense multiply-accumulates by the conventional count; other floating-point
operations; transcendentals; every required input byte read once; every
output byte written once. `bound.demand_of(module, inputs)` measures it from
the case itself, the way the evaluation checks a point's declared demand:
MACs from PyTorch's FLOP counter, bytes from the real tensors. The counter
sees products and convolutions, not elementwise arithmetic, so for a case
whose work is elementwise pass `FP_OPS` and `SFU_OPS` yourself, or declare a
`bound.Demand` by hand. Undercounting is safe: the bound only gets looser and
η lower. Overcounting is not: η > 1 is an audit failure in the evaluation,
never a good result.

**Capacity** is `spec_peak`: what the profile's configuration can serve per
cycle, read from its target file (vector length, issue width, whether it has
an SFU) and from RTL facts (one memory port of 8 bytes per cycle, one SFU
slice). Work several units can do is divided by their summed rate, so the
bound assumes no algorithm. If your design changes a hardware fact that
`scripts/bound.py` holds as a constant (`MEM_DW_BITS`, `SFU_SLICES`), change it there
too; otherwise the bound you see is the base machine's.

A profile is one core, so L and η are **one core's**, exactly as the
evaluation's own bound is; the script says so beside every L. On a machine
with several cores of the profile, η still compares cases with each other
and a version with the one before, which is what you need it for; it is
not how close that machine is to its own limit. How you stand against the
base machine is the ratio of its cycles to yours on the same case, so
measure the base before you change it and keep those numbers (`--json`).

Read the two regimes apart. A case bound by `fp_lane` has its arithmetic as
the limit: the compiler's schedule, the matrix unit's utilisation, the
pipeline. A case bound by `mem_bytes` has the memory port as the limit: how
many bytes the program really moves against how many it must. A change that
helps one regime can do nothing for the other, and a benchmark built on this
method has cases in both.

## 4. Designing the set

Start from the class the task names. Ask what the class varies in, and write
one small case per answer, so that a change is measured on each dimension
and not on the one shape you happened to try first. The dimensions that make
a compiler or a machine behave differently are few and general:

| Dimension | Why it is a dimension |
|---|---|
| dtype | FP16 and FP32 take different arithmetic units and different tolerances; a lowering that is right in one can be wrong, or slow, in the other |
| the leading dimension | one row (M = 1) against a matrix reuses nothing, so the memory port binds; many rows reuse the weights, so arithmetic binds. A method that helps one can hurt the other |
| a batch dimension | independent problems in one call: how launches are formed, whether the batch is a loop or a grid |
| an epilogue | a bias, an activation, a residual after the main operation: whether it fuses into the kernel or costs a second pass over the output |
| a tail | a dimension that is not a multiple of the vector length: the masked last block takes a different path, and a change that is right for full blocks can break it |
| the reduction length | a long K against a short one changes where the time goes between loading operands and accumulating |
| the stream length | how many bytes one launch streams through the memory path once, against what a core keeps in flight: a short stream hides the memory's round trip, a long one is paced by how many requests stay outstanding against it. One row against a large matrix is the decode-time case; it is where a machine's memory-level parallelism shows, and a change there is invisible on a small matrix |
| a cached context | one query against a context computed earlier: the stream is the stored context, not the weights, it grows with the context, and the reduction ends in a normalisation. A decode step is made of these as much as of projections |
| several launches | a real program is a graph, not a kernel; the time between launches, and what each launch's inputs cost to move, only show in a multi-launch case |
| input values | large values, values near a tie, NaN and infinities where the operator defines them: the protocol compares NaN positions, and a numerically careless change passes random inputs and fails these |

Then pick sizes. One size the simulator finishes in seconds is enough for
correctness and for finding where the time goes; a larger one tells you
whether a gain holds when the data no longer fits where the small case kept
it. Shapes are yours to choose, and an arbitrary choice is a better test than
a round one. A set of six to ten cases built this way covers a class; a
single shape covers nothing but itself.

**A worked example** for the matrix-multiply class is in `examples/`: eight
cases, one per dimension, with a note on what each isolates; the last two
are the decode-time shapes (one row against a large matrix, one query
against a cached context), which a class of model workloads always has and a
set of small square shapes never covers. Copy the pattern for any class.

## 5. Measuring

Build the device as the task text says, with the environment it says
selects it, then

    python /opt/tnca-skills/testcase-design/scripts/tnca_case.py CASE.py [...] --profile <the profile you are scored on>

Run it from your working directory (the repository checkout): the bound reads
that checkout's `spec/` and `compiler/targets`. The script sets `TNCA_PROFILE`
and `TNCA_OPT_LEVEL` and nothing else about the machine: the backend and its
arguments are your shell's, as the task text sets them, so the device it
times is the one your own Python would.

One line per case: status, cycles, launches, the error against the
tolerance, η and the binding resource; the problems below it when it did not
pass. `--json FILE` appends the full record of every case, so you can keep
before-and-after numbers across your changes. From Python,
`tnca_case.measure(module, inputs, profile=...)` returns the same record.

A case file is a module with `make() -> (module, inputs)` and optionally
`MODEL = True` (a whole model, judged at the model tolerance), `TOL`,
`FP_OPS`, `SFU_OPS`, or a `DEMAND` declared by hand. See `examples/`.

## 6. What this does not tell you

It does not tell you the benchmark's programs, shapes or sizes, and nothing
here is derived from them. The method is the benchmark's; the cases are
yours. A change that is good on cases built this way, across the dimensions
of the class, is what the evaluation is designed to reward; a change that is
good on one shape is what it is designed not to.
