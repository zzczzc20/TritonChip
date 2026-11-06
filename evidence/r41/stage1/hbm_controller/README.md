# The B40 image's HBM controller, and the slow DRAM path checked against it

## Source

`a3_hbm_inst.xci.gz` is the HBM IP configuration of the image a88ebdf5's own link project:
- the attempt-3 build on the FPGA host,
  `xo/tmp/link/vivado/vpl/prj/prj.gen/my_rm/bd/ulp/ip/ulp_hmss_0_0/bd_0/ip/ip_0/bd_85ad_hbm_inst_0.xci`;
- inside `xilinx.com:ip:hbm_memory_subsystem:2.0` (`hmss_0`);
- sha256 of the uncompressed file: `e45a39027f01aefd0d3294a99880bad9701f2545886f003996cf62d14e8f2502`.

It was read with grep only; nothing was rebuilt.

Settings, identical for every memory controller (MC0–MC15) checked:

| Setting | Value | Slow-path model (`ramulator/hbm2_u280_pc.yaml`, bridge) |
|---|---|---|
| `USER_MC0_PRE_DEF_ADDR_MAP_SEL` | `ROW_BANK_COLUMN` | now the same (below) |
| `USER_MC0_BG_INTERLEAVE_EN` | `true` | now the same |
| `USER_MC0_ADDR_BIT` (MSB..LSB above the 32 B burst) | `NA,RA13..RA0,BG1,BA1,BA0,CA5..CA1,BG0` | now the same |
| `USER_MC0_REORDER_EN`, `REORDER_QUEUE_EN` | `true`, `true` | FR-FCFS scheduler, separate 32-entry read and write buffers: a different abstraction, not shown equivalent |
| `USER_MC0_LOOKAHEAD_ACT`, `LOOKAHEAD_PCH` | `true`, `true` | not modelled as such |
| `USER_MC0_CLOSE_PAGE_REORDER` | `false` | Open row policy |
| `USER_MC0_WRITE_READ_REGROUPING` | `false` | watermark write mode (0.2/0.8) |
| `USER_MC0_REF_CMD_PERIOD` | `0x0DB6` = 3,510 | nREFI 3,510 (all-bank refresh) |
| `USER_MC0_ENABLE_ECC_CORRECTION`, `ENABLE_ECC_SCRUBBING` | `true`, `true` | not modelled (scrub traffic absent) |
| `USER_HBM_TCK_0` | 900 MHz (1.111 ns) | tCK 1,111 ps |
| `USER_HBM_DENSITY`, `USER_HBM_STACK` | 8 GB, 2 | 32 PCs; the B40 window exposes PC0–15 |
| `USER_AXI_CLK_FREQ` | 450 MHz | outside the DRAM model, folded into `dram_fixed_cycles` |

## Address map: corrected to the controller's

`USER_MC0_ADDR_BIT` puts the bits above the 32-byte burst offset in this order:
- BG0 first;
- then the column (CA1–CA5);
- then BA0–BA1;
- then BG1;
- then the row.

The bridge's earlier layout was a synthetic "bank groups and banks first" interleave. Its source comment said so, and
`address_map.md` declared it unverified.

| Sequential stream | Controller (now) | Earlier bridge |
|---|---|---|
| A 64 B line | spans BG0 = 0/1 | spans two bank groups |
| Stays in one bank pair for | 2 KiB (32 columns) | 128 B |
| Banks touched per 8 KiB | 4 (bank pairs) | all 16 |

Row-hit and conflict behaviour of concurrent streams in one PC depends directly on this. Results from the earlier
layout keep their own source identity and are not reinterpreted.

`tnca_dram_bridge.cpp: map_address` now implements the controller bit order. `bridge_test.cpp`'s new
`controller_address_map` test checks the decoded fields for every burst of 32 KiB against `USER_MC0_ADDR_BIT`, and the
2 KiB, 8 KiB and 16 KiB boundaries.

## Two exactness defects in the replay's tick skipping, found and fixed

The replay skips ticks for speed. It is meant to be cycle-identical to ticking Ramulator2 every cycle.

1. **Write-mode evaluation lost.**
   - Each skipped tick of a busy controller reaches `pick_rw_if`, which re-evaluates the read/write mode while the
     priority buffer is empty.
   - Skipping those ticks lost a mode switch that was due just before a refresh entered the priority buffer. The run
     then stayed in write mode through the refresh and reordered what followed.
   - Found with the new map: 7 of 200,000 completions differed between busy-skip and no-skip. The command logs are in
     `bridge-v2-evidence.tgz:skipdbg.cmd`, made with the diagnostic in `skip_defect/`.
   - Fix: `skip_to` evaluates the mode once. The buffers do not change while ticks are skipped, and `set_write_mode`
     reaches its fixed point in one call.
2. **Refreshes in an idle gap merged.**
   - The idle skip jumped to the last refresh due in a gap.
   - Ticking every cycle, the first refresh of a gap closes the open rows, and the later ones issue REFab at once.
   - The merged refresh found the rows still open and paid tRP (13 cycles) before REFab, delaying everything after it.
   - Found by the new reference: 3,565 of 200,000 completions differed from every-cycle ticking.
   - Fix: stop at every refresh due in a gap. That is a few ticks each, because the busy skip covers the refresh's own
     wait.

`replay(..., event_skip, idle_skip)` gains `idle_skip=false` for the every-cycle reference. `event_skip_identical`
now compares every-cycle ticking with idle and busy skipping on the 200,000-request mixed trace (streams, random
lines, writes, bursts, idle gaps across refreshes):
- 0 differing completions;
- equal row conflicts;
- 73,173,703 ticks without skipping, 1,744,779 with;
- all 20,847 refreshes are now modelled.

These defects affected every slow-path result before this change, including the harness smoke v5/v6 slow graphs and
the fragment and prospective SystemC runs on the slow path. Those results keep their source identity. The fast path is
unaffected.

## Results (`bridge-v2-evidence.tgz`, GB10, 2026-10-21)

- **Bridge tests (10/10 pass):** idle_read, row_hit_miss_conflict, controller_address_map, bank_parallelism,
  refresh_on_stream, idle_skip_with_refresh, write, write_queue_backpressure, read_write_turnaround,
  event_skip_identical.
- `tnca-dram-path-test` passes (configs/r40-board-m3.cfg).
- `tnca-timeline-drain-test` passes (configs/r40-board-m2.cfg).
- Full model build: tnca-chip, tnca-tlm and the tests, with Ramulator2 at the pinned c6c7a5de.

`sources.sha256` records the bridge sources these results come from.

## What this does not establish

- The FPGA controller's reorder queue, lookahead activate/precharge, AXI queueing and ECC scrubbing are not modelled
  as such. Equivalence of queueing behaviour is not claimed.
- `dram_fixed_cycles` (20 core cycles), which stands for the AXI switch, the controller pipeline and the PHY, is still
  a fitted constant (see `latency_boundaries.md`).
- No board timing is claimed; the model remains uncalibrated.
