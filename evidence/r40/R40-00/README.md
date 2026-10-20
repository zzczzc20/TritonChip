# R40-00: the protocol frozen, B39 reproduced, 2026-10-10

`make r40-protocol`, `make r40-freeze-check` and `make r40-b39` on the build
host (GB10), judged by `tools/r40/acceptance.py --wp R40-00`: every gate
`capability-pass` (`gates.json`, the evidence file each tool wrote).

| Gate | What it checked | Result |
|---|---|---|
| R40-00.protocol | `tools/r40/protocol_tests.py`: every manipulation of a comparison refused for the right reason, identical inputs give S_geo 1, `acceptance.py` passes nothing missing | 34/34 |
| R40-00.frozen | `tools/r40/freeze.py --check`: plan, 21 subsets, ablation matrix, ArchBench suites (5 tiers), API lock, B39 record hashes, HBM coverage record | 33/33 |
| R40-00.b39-gates | `tools/r40/b39_gates.py`: R39's 27 merge gates on this tree against `spec/r40/b39/gates.json` (exit and counts) | 27/27 |
| R40-00.b39-reproducible | `tools/r40/b39_repro.py`: B39 rerun, protocol 2 on the same image, the identity through `ablate.py`, the gate run | all |

## The board runs (U280, image u280_r39h, 100 MHz)

- **B39 rerun:** an export of `be87779`, protocol 1, as its records were made.
  77/77 pass; every point's device cycles equal to `spec/r40/b39/<tier>.jsonl`.
- **Protocol 2:** the R40-00 tree, 5 warm-up and 10 timed calls per point,
  provenance bundles.
  - 77/77 pass. First-call cycles are equal to B39's, with no spread across the timed calls.
  - Time boundaries are consistent: host run ≥ device time, request ≥ each of its parts, first call ≥ a steady call.
  - Compilation: median 3.5 s per point.
  - Device time is a median 0.3 % of a request on this CPU-tensor path.
- **Identity:** B39 against that run through `tools/r40/ablate.py`. Only the software tree differs, and it is declared. Accepted, S_geo 1.000000, CI [1, 1], verdict no-significant-gain.

The records are under `sim/build/r40/b39_rerun/` and `sim/build/r40/b39_p2/`
on the build host (not in git). Their summaries are above and in `gates.json`.

## The gates that do not exit 0 on B39 either

Frozen with their exit status and reason (`spec/r40/b39/gates.json`,
`why_nonzero`). A later run must give the same exit and counts:

- `r38-l0`: two checks pending by design (S1-04's degraded builds, and an `r38-baseline --run`); 16 pass, 0 fail.
- `r38-baseline-check`: the R38 baseline was frozen with R38's profile configuration, and R39 changed the profiles. All 149 cases and 45/45 operator/level pairs pass.
- `r38-identity`: needs `make r38-baseline` first. Artifact identity is `r38-torch`'s plugin identity, 149/149 at O0/O1/O2.
