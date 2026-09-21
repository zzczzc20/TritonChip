# R3.6: where it stands, and the one thing still open

Written as a handoff. It says what is done, what is not, and — for the
single unresolved failure — exactly what is known, what has been ruled
out, and what to do next. Nothing here is a claim that the open item is
harmless.

## The state

30 standards, 11 checkers, all wired into `make`. On the locked host
(`spec/r1_toolchain.lock.json`'s platform), the gate's last run gave:

| package | result |
|---|---|
| R36-01 plan | 3/3 |
| R36-02 backend | 3/3 |
| R36-03 config | 3/3 |
| R36-04 rtl | 3/3 |
| R36-05 memory | 3/3 |
| R36-06 io | 3/3 |
| R36-07 build | 3/3 |
| R36-08 conformance | 1/1 |
| R36-08 mutation | 1/1 (7 mutations, 4 clean controls) |
| **R36-08 regression** | **FAIL** — see below |
| **R36-09 handoff** | **FAIL** — see below, likely fixed |

Verdict: `NOT ACCEPTED`. `make r36-acceptance` reproduces it;
`make r36-verify-evidence EVIDENCE=<dir>` re-derives the verdict from
the raw runs.

Section 5.2's fidelity **passes**: the six frozen R3.5 workloads produce
byte-identical machine code at B0 and O2 against the frozen baseline
`0e7c34795cd6`, and all 11 cycle differences are attributed in
`spec/r36_baseline_delta.json`.

## The open failure

`REG-profiles` (R1's `r1-profile-tests`, case `CFG-01`) times out on the
banked-VRF bench:

    core-vrf-sync-v1: 10 facts on sim/build/tb_core_vrf.vvp -> FAIL
    FAIL: timeout after 200000 cycles

The program is ten `s.rdspecial` / `.assert_eq` pairs and an `exit` —
a few hundred cycles of work.

### What is known

- It **passes standalone**, 10 runs out of 10 on the same host, same
  clone, same binary.
- It **fails inside the gate**, twice, reproducibly at that position.
- `R36-HAND-01` has the same shape — fine alone, broken in the gate —
  and its error was `Code generator failure: -1` from `device.prepare`
  building into a directory that had just been created.

### What has been ruled out

- **Not a hardware race.** iverilog is deterministic on identical
  inputs, and the case passed 10/10 in isolation.
- **Not the missing bench.** `r1-profile-tests` never declared
  `tb_core_vrf.vvp` as a prerequisite — fixed in `12c4fe3` — and the
  failure survived the fix, now as a timeout rather than "not built".
- **Not disk space.** The root filesystem held at 5.4 GiB free
  throughout; the gate's scratch is on the mounted disk and peaked at
  100 MiB.
- **Not the workload caps or the budget**, both of which are asserted
  and within limits.

### What was fixed along the way, and may have fixed it

`device.prepare` did `makedirs` → `prune` → build, so it pruned between
creating its output directory and writing into it. And `prune` matched
profiles by prefix, so `core-sim-` also matched `core-sim-slow-…` and
preparing one profile could evict another's builds. Both fixed in
`fc6e3cb`. **The gate has not been re-run since.** That is the first
thing to do.

### If it still fails

The inputs are now preserved and the reproduction is one command. The
gate keeps its scratch directory when the verdict is not a sign-off
(`b882874`), `profile_tests` prints where it left the assembly and hex
and the exact `vvp` line to re-run them, and it logs the bench's
`cycles=N issued=M …` line instead of iverilog's `$finish` (`1a13308`).

**Read `issued=` first.** `issued=0` means the program never executed an
instruction — a bad image or a memory never loaded, which points at the
hex or at `u_mem.load_hex`. Non-zero means it genuinely ran and hung,
which points at the core.

Then diff the preserved `cfg01-core-vrf-sync-v1.hex` against one from a
passing standalone run. The simulation is deterministic, so if the
inputs match, the binary differs; if the inputs differ, the target JSON
or the assembler did.

## Coverage that is knowingly partial

`spec/r36_required_cases.json` → `known_partial_coverage` records four,
each with what is not checked, why, and what would close it:
`R36-RTL-03` (proc/check scoped to the shell), `R36-EV-02` (evidence per
case, not per sub-clause), `R36-CFG-02` (no artifact needing an absent
capability), `R36-REL-01` (the regression runs once, not twice).
`R36-BASE-01` asserts that record is present and complete, so the list
cannot quietly empty itself.

## Two inherited items, open in the baseline too

- **R35-OPT-02** — the pinned Triton wheel exports no AxisInfo and the
  TTGIR coalesce route does not surface its conclusion. Both routes are
  recorded in the checker. Not closed here, and not claimed to be.
- **CAP-TWO-ROLE** — the core has two role contexts and the product path
  cannot reach them: `runtime.py` launches with `role_mask=1`, hardcoded,
  and the GEMM builder emits a single-role tile. Section 5.1's two-role
  leg is recorded NOT RUN.

## What a reviewer should distrust

Three defects in this stage's own tooling made a simple failure take
three gate runs to corner, and all three were about discarding evidence
rather than about the code under test:

1. the regression runner kept the last four lines of a failing suite,
   and a suite prints its verdicts at the end and its reasons in the
   middle;
2. the gate deleted its scratch directory, which is where a failing
   checker leaves its inputs;
3. `CFG-01` logged the last line of a failing simulation, which is
   `$finish called at …` — identical whatever happened.

All three are fixed. The lesson generalises: when adding a checker here,
make sure a failure of it can be diagnosed without re-running it.
