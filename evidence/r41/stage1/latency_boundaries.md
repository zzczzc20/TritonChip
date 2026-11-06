# Memory latency boundaries of the two DRAM paths (no double counting)

This audits contract S1-01 C: the boundaries between AXI, NoC, home engine, controller and DRAM have to be explicit,
and the detailed path must not re-count delays the fast model already includes. All source references are to
`tools/performance_harness/systemc/`.

A word access from a core on the timeline memory (`include/tnca_timeline_memory.hpp`) passes through these segments
in order. The two DRAM paths differ in exactly one call, `DramPath::complete` (`include/tnca_dram.hpp`). Everything
before and after it is the same code in both paths.

| # | Segment | Owner (both paths unless stated) | Config keys |
|---:|---|---|---|
| 1 | Core → home over the NoC, request VN | `route()` link timelines | `tl_noc_inject_cycles`, `noc_stages` |
| 2 | Home slice lookup / hit / miss / MSHR / victim | `slice()` engine timeline | `tl_slice_*` |
| 3 | Slice → outlet | `slice()` | `tl_slice_to_outlet_cycles` |
| 4 | Outlet admission (TNCA `tnca_line_axi` outstanding window) | `line()` inflight list | `outstanding` |
| 5 | Outlet issue stage, 2 cycles (read) or 3 (write) | `line()` issue timeline | `tl_outlet_read_issue_cycles`, `tl_outlet_write_issue_cycles` |
| 6 | **HBM side: AXI switch + HBM controller pipeline + PHY + DRAM** | **`DramPath::complete`** | see below |
| 7 | Second 32 B AXI R beat into TNCA (reads) | added after `complete` in both paths | `tl_outlet_beat_cycles` |
| 8 | Outlet → slice, response return | `slice()` | `tl_outlet_to_slice_cycles` |
| 9 | Fill / wake | `slice()` engine timeline | `tl_slice_fill_occupancy`, `tl_slice_wake_occupancy` |
| 10 | Home → core over the NoC, response VN | `route()` | `tl_noc_eject_cycles`, `noc_stages` |

Segment 6 in each path:

| | Fast (`FastDram`) | Slow (`SlowDram` + Ramulator2 replay) |
|---|---|---|
| Fixed part (AXI switch, controller pipeline, PHY) | inside `hbm_latency_cycles` (23) | `dram_fixed_cycles` (20) |
| Controller queueing and scheduling | not modelled | Ramulator2 FR-FCFS, open page, 32+32 buffers, write watermarks |
| DRAM device timing (tRCD, tCL, bursts, tRP on conflict, refresh) | inside `hbm_latency_cycles` | Ramulator2 HBM2 at 900 MHz, `ceil(latency / dram_clock_ratio)` |
| Write | `tl_hbm_write_latency_cycles` (23) | 20 + ceil(WR command + CWL + BL from the replay / 10) |

Why the slow path counts nothing twice:

1. **Segments 1–5 and 7–10 are outside `complete`.** In the slow path they are computed by the same code from the same
   keys, so the replacement of segment 6 cannot overlap them. `src/dram_path_test.cpp` (`fast-boundaries`,
   `slow-fallback`) checks the call boundary. `tnca-dram-path-test` passes in `hbm_controller/bridge-v2-evidence.tgz`.
2. **Ramulator2 contains no fixed pipeline term.** Its latency runs from `send` to the burst callbacks, which is
   scheduling plus device timing only. The bridge test `idle_read` measures an idle 64 B line read at 34 DRAM cycles,
   against tRCD + tCL + tBL = 28 for the first burst. The second burst and command scheduling make up the rest, and the
   test bounds the excess at 8 cycles.
3. **The fixed part is counted once.** It is `dram_fixed_cycles` in the slow path and part of `hbm_latency_cycles` in
   the fast path, never both.
   - The fast idle read is 23 + 1 (beat) core cycles.
   - The slow idle read is 20 + ceil(34/10) = 24, + 1 (beat).

   The two agree to within one core cycle on an idle line. That cycle is the second DRAM burst, which the fast
   constant folds in.
4. **The second burst is not the second beat.** The DRAM-side second burst (two 32 B BL4 bursts per line, inside
   Ramulator2's completion) and the TNCA-side second AXI beat (segment 7, `tl_outlet_beat_cycles`) are different
   transfers on different interfaces. Each is counted once.
5. **Admission and controller queueing are distinct, and do not overlap in practice.**
   - TNCA's outlet window (segment 4) is at most `outstanding` lines per outlet: 8 on the board.
   - Ramulator2's read/write buffers hold 32 bursts each. One outlet's 8 lines (16 bursts) cannot fill them, so its
     requests never wait in both queues for the same reason.
   - Several outlets mapping to one PC can still queue in Ramulator2, which is the intended effect.
6. **Host maintenance traffic is a separate domain.** It is accounted outside launch windows; see `memory_paths.md`.

What remains assumed:
- `dram_fixed_cycles` = 20 is the M2 fast latency (23) minus the idle DRAM part (3). It is a constant, not measured on
  the board. A board microbenchmark separating the AXI/controller pipeline from DRAM timing is a Stage 2 candidate.
- The FPGA controller's reorder queue and lookahead ACT/PCH (`hbm_controller/README.md`) are not modelled as such.
- The model stays uncalibrated. This audit establishes the boundaries, not accuracy.
