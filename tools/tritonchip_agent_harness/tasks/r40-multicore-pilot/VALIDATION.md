# R40 multicore pilot: provenance and validation

This is a **pilot task**: the first task on a multicore machine. Its base,
`6b36db5a06…` (the full hash is in `task.json`), is the PR9 harness branch
with `main` at `3e1a282` (R40, PR #10) merged in. The checks below up to the
known-answer submissions ran on the earlier base `83e59b8`; between the two
nothing under `rtl/` or `tb/` changed (the mailbox was split by direction
and the review's fixes landed, see the last section). The machine is R40's three-core
U280 machine as `spec/devices/u280-3core-r40.json` describes it: three
`core-vrf-sync-v5` cores, a NUCA home slice per core, three 256-bit HBM
outlets, 90 MHz. The description generates the 39 parameters of R40's
delivered image (`a88ebdf5`, attempt 3) and states the two that image left at
the top's defaults.

The six points are tiny ArchBench ops chosen on the base's three-core
simulation, each spread over the cores: grids of 4 to 63 programs, one op of
three launches. A point takes one to six minutes to simulate.

## How a point is measured

Each point runs against a device of its own: a container that builds the
image's top level in the board-register bench from the candidate's
description (base code, the candidate's hardware only) and serves it through
a mailbox; the host container runs ArchBench with the candidate's compiler
and reaches the device only through that mailbox. The score is the bench's
observer: each launch's cycles between the control block's MMIO endpoints,
written where the host cannot write. The host's record must equal the
device's own done count, no launch may fault, and the device must have
drained when it reports a launch done.

## Checks

### Determinism (2026-10-27)

The bench's AXI line models stall at random from a generator that advances
every cycle, so before `97fedf5` a launch's cycles depended on how long the
host took before it: the same point counted 4343 cycles through a direct
link and 4371 through a mailbox, and 3559 or 3504 from one run to the next.
With a description the bench reseeds the line models as each launch starts.
op.add/tiny then counted 4412 cycles through a direct link, a mailbox, and a
mailbox whose host had a fifth of a CPU; op.softmax/tiny 3559 all three ways;
the observer's lines were identical.

### The base (2026-10-27)

Every point passed; each launch drained at once (settle 0 over all eight
launches), and the observer's count is the device's done count plus the two
cycles of the MMIO endpoints, per launch:

| Point | Launches | Observer | Device's count |
|---|---|---|---|
| op.add/tiny/fp16-n1000 | 1 | 4414 | 4412 |
| op.layernorm/tiny/fp16-r4-d64 | 1 | 3814 | 3812 |
| op.linear/tiny/fp16-m16-k64-n64 | 1 | 9628 | 9626 |
| op.transpose/tiny/fp32-r32-c32 | 1 | 3621 | 3619 |
| op.softmax/tiny/fp16-r4-c64 | 1 | 3561 | 3559 |
| op.rope/tiny/fp16-b1-h2-s16-d16 | 3 | 5992 | 5986 |

### The leak audit (2026-10-27)

An export of the base with this task's hide list held 706 files. None
contained a point ID, nor any point's shape suffix (`fp16-n1000`, `r4-d64`,
`m16-k64-n64`, `r32-c32`, `r4-c64`, `b1-h2-s16-d16` and their shorter forms).

### The base unchanged (`--agent noop`, 2026-10-27)

`run-task ... --agent noop` on the Docker test host: `validity=valid`, gate
`PASS`, validation **3/3** and final **3/3** passing, every speedup exactly
**1.000** -- each point counted the same cycles as the base's measurement
made separately before it (the table above). Run ID
`r40-multicore-pilot-noop-20261026T174818Z-741e`; `run.json` SHA-256
`c5c74e2fefc8e4a8c8c4d4b0abd82e3fbe2b7ddd78fc051cc6f476e8601d2f4f`; toolchain
image `sha256:88bc6930ffef777487dc568b3349ddcdc487b2905f6812fad02a7be5f93966e4`.
The first attempt (`...-3747`) failed every point in eight seconds on a
harness bug (the host's timeout passed twice to the runtime), fixed in
`7e85ce2`.

### Known-answer submissions (`--agent patch:known/*.patch`, 2026-10-27)

Each changes one field of the description's uncore. Every run was `valid`,
the gate took exactly `spec/devices/u280-3core-r40.json`, the record marks
the hardware changed and certification pending, and every point passed:

| Point | Base | `SETS` 32→4 | `AXI_OUT_N` 8→1 | `MEM_PORTS` 3→1 |
|---|---|---|---|---|
| op.add/tiny/fp16-n1000 | 4414 | 4449 (0.992) | 4475 (0.986) | 3855 (1.145) |
| op.layernorm/tiny/fp16-r4-d64 | 3814 | 3813 (1.000) | 3814 (1.000) | 3657 (1.043) |
| op.linear/tiny/fp16-m16-k64-n64 | 9628 | 11617 (0.829) | 11286 (0.853) | 8997 (1.070) |
| op.transpose/tiny/fp32-r32-c32 | 3621 | 3631 (0.997) | 4163 (0.870) | 3331 (1.087) |
| op.softmax/tiny/fp16-r4-c64 | 3561 | 3561 (1.000) | 3654 (0.975) | 3348 (1.064) |
| op.rope/tiny/fp16-b1-h2-s16-d16 | 5992 | 5991 (1.000) | 6259 (0.957) | 5491 (1.091) |
| geomean, validation / final | | 0.937 / 0.999 | 0.944 / 0.933 | 1.085 / 1.081 |

Cycles from the observer, the speedup against the base in brackets. Run IDs
`r40-multicore-pilot-patch-sets-4-20261026T175839Z-34f5`,
`...-outstanding-1-20261026T180150Z-4c55`, `...-mem-ports-1-20261026T175531Z-579b`.

- **A cache of four sets** slows the matrix multiply, the point that reuses
  its operands (0.83), and leaves the streaming and row-wise points within a
  percent: as expected.
- **One outstanding request per outlet** slows the points with memory-level
  parallelism (the matrix multiply 0.85, the transpose 0.87, the three-launch
  op 0.96): as expected.
- **One outlet makes every point faster** (1.04–1.15), against the
  expectation. The bench gives each outlet its own latency, 25 + 19·p cycles
  (25, 44 and 63 for outlets 0–2: farther from the HBM, slower), and these
  points are latency-bound, so routing all three cores through outlet 0 wins.
  On the card R40 measured the opposite small effect: three outlets 1.007×
  one (R40-06.ports, [0.998, 1.025]), the home-slice engine and not the
  outlets being the limit. The simulation and the card therefore disagree on
  this field's direction; certification would catch a candidate that relies
  on it, but multicore certification is not wired yet. **To be decided before
  an agent runs:** fix `MEM_PORTS` for this task, or calibrate the bench's
  outlet latencies against the card.

Every launch of every run drained at once (settle 0 on all 24 launches), as
on the base. `drain_cycles` is frozen at **8**: the base's largest settle (0)
plus eight cycles for a legitimate register stage on the completion path.

### After the independent review (base `6b36db5`, 2026-10-27)

The review's fixes changed how a point is run, not the machine: the mailbox
is split by direction, no container outlives an evaluation, a multicore base
is kept per machine. On the new base the base's measurement counted exactly
the cycles of the table above, point by point, and `run-task ... --agent noop`
was again `valid`, gate `PASS`, validation **3/3** and final **3/3**, every
speedup exactly **1.000**, with no evaluation container left behind. Run ID
`r40-multicore-pilot-noop-20261026T182221Z-f124`; `run.json` SHA-256
`83025a485c011728778e837c75e0e3cf3927ff98b4148287d0b857b625c0b557`. The
known-answer submissions were not repeated: nothing they measure changed.
