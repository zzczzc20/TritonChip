# Routed census of attempt 3 at 100 MHz

Read-only census of attempt 3's routed checkpoint (54249f7f, image a88ebdf5,
`level0_wrapper_routed.dcp`, kernel clock 10 ns). It ran on the FPGA host
from 2026-10-19 20:29 to 21:11 UTC with `tools/r40/census_failing_endpoints.tcl`.
The card was not touched. This is the first gate of the final-implementation
plan ([`../../final_implementation_plan/README.md`](../../final_implementation_plan/README.md)
§2.3, §2.5).

## Result

| Population | Failing endpoints | TNS (ns) | Worst slack (ns) |
|---|---:|---:|---:|
| All of attempt 3 | 17,332 | -6,679.0 | -1.002 |
| Residual: what-if exclusion of the M1–M3 cone | **12,799** | **-4,110.8** | **-0.919** |

**The launch rule in §2.5 is not met.** That rule asks for residual WNS ≥ 0,
or a committed fix with unit OOC WNS ≥ +1.0 ns for every residual group. What
remains after M1–M3 is not one more cone. It is 753 path families over 94
source/destination unit pairs, almost all inside the tiles. With M1–M4 alone,
the expected link result is about **91 MHz** (1000 / 10.919). How many
families block each target clock is in [`frontier.json`](frontier.json):

| Target | Families to fix | Unit pairs | Endpoints |
|---:|---:|---:|---:|
| 92 MHz | 18 | 14 | 2,707 |
| 93 MHz | 89 | 58 | 5,874 |
| 95 MHz | 170 | 83 | 8,959 |
| 100 MHz | 753 | 94 | 12,799 |

## Structure (whole census, from `report_design_analysis`)

| Class | Endpoints | TNS (ns) |
|---|---:|---:|
| SLR crossing: the M1–M3 cone and its neighbours | 4,717 | -2,640 |
| Through a placer-inserted BUFG: the core's `stall_front`, the clock enable of the X-stage operand registers `x_v{a,b,c}_q` and `of_d*` (fanout 1,653) | 4,650 | -2,168 |
| Other logic | 3,200 | -745 |
| To or from RAM | 2,960 | -739 |
| Through DSP | 1,805 | -388 |

- **The BUFG class sets TNS, not WNS.** It carries the core's whole stall
  decision: the X-stage scalar read feeds `u_srf`, then the barrier test's 48-bit
  compare, `adv`/`d_valid`, and finally `stall_front`, a 20-level path. About
  3 ns of each of these paths is the trip to and through the BUFG. Fixing it
  removes about half the residual TNS, but WNS stays at about -0.92 ns.
- **The paths that set WNS** sit at -0.87 to -0.92 ns in about ten independent
  structures:
  - VFPU `red_busy` into the FMA stage-0 inputs;
  - the X-stage scalar read into the scoreboard and the MTX inputs;
  - writeback `w_rctx` into `w_sdata`;
  - the scoreboard `sb_v` into the VRF bank `s1_vld`;
  - the dispatch DSP `total` into `remaining`;
  - the slice `cur_ms` into `data_mem` (C1, 658c811);
  - XLS FMA stage balance (`p0_sticky_al` into `p1_sh_mag`, and `p1_sh_mag`
    into the MTX accumulator lane memory);
  - LSU `pending` into the mesh input FIFO.

  The list per source is `blocking_95mhz_by_source` in `frontier.json`. Most of
  these fixes add a pipeline stage, which changes cycle behaviour. Each one
  needs full regression, not just a cycle-equivalence check.

## Files

- `census.tgz`: raw census output. It contains `summary.tsv`, `failing_endpoints.tsv`,
  `residual_failing_endpoints.tsv`, `logic_levels.rpt`, `design_analysis.rpt`
  (20,000 paths), `residual_top_paths_full.rpt` and `top_paths_full.rpt`.
- `residual_groups.json.gz`: `tools/r40/group_failing_paths.py --tsv
  residual_failing_endpoints.tsv --tns -4110.8 --endpoints 12799`. Coverage is
  1.0 of the endpoints and of the TNS.
- `frontier.json`: the per-target frontier, the structural classes above, and
  the sources that block 95 MHz.
