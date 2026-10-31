# Wider tag benches (G4 step 2)

**Domain: RTL simulation (Icarus Verilog) on GB10.** Functional results only: no timing, resource, bandwidth or board claim. No card was accessed.

## What changed
`HOME_MSHR_N` and `LINE_SLOTS` (G4_DESIGN.md §8.2) widen the memory and line tags beyond the four bits the existing unit benches were written for. The benches now take the width as a parameter, and their drivers sweep it:

| Driver | Bench | Sweep |
|---|---|---|
| `tools/r41/lsu_line_tests.py` | `tb/core/tb_lsu_line.v` | line LSU slots up to 64; 48 configurations |
| `tools/r41/wide_noc_tests.py` | `tb/noc/tb_wide_ni.v` | NI `TAG_W` up to 7 (722-bit payload); 12 configurations |
| `tools/r41/lsu_hybrid_tests.py` | `tb/core/tb_lsu_hybrid.v` | hybrid LSU `BEAT_W` 4–7 × 3 seeds × port stall 0/30 % × burst off/on; 48 configurations |

The default parameters of every bench are the configurations that existed before, which still run as part of each sweep.

The bench and driver changes were written by the Kimi Code agent (tmux session 8). Claude reviewed them and corrected the hybrid bench after two failed attempts, which are kept.

## Attempts (hybrid LSU)
1. `hybrid_attempt1_timeout.log`: `BEAT_W` 6 and 7 with `MAXLAT` 1024 ran into the bench's hang bound (`100*NACC+10000` cycles). This was not a deadlock: a longer run passes (275,760 cycles against the bound of 70,000). The bound now scales with the latency, `(MAXLAT>64 ? 4*MAXLAT : 100)*NACC+10000`.
2. `hybrid_attempt2_coverage.log`: 42 of 48 pass. The six failures are `BEAT_W` 7 with bursts. The miter and memory comparisons passed, but the coverage check failed: the fast-slot peak reached 56–58 of 64. The known store/CAS chain that the bench inserts every 64 accesses broke every burst before 64 lines could be in flight. The chain period is now a parameter (`CHAIN_PERIOD`, default 64; 256 for `BEAT_W` 7, with a 256-access burst period). The driver also records a `vvp` timeout as a failure instead of aborting.
3. `hybrid_attempt3.log`: **48 of 48 pass**, with a fast-slot peak of 64 at `BEAT_W` 7. The header of the log gives the commit and the hash of the working-tree diff it ran on.

`lsu_line.log` (48 of 48) and `wide_noc.log` (12 of 12) passed on the first run.

## Commands
```sh
python3 tools/r41/lsu_line_tests.py
python3 tools/r41/wide_noc_tests.py
python3 tools/r41/lsu_hybrid_tests.py
```
