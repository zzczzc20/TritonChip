# Final-image candidate: M1-M4 on branch r40-final-image

This is the hardware candidate for the last full U280 implementation planned
in `evidence/r40/v3/final_implementation_plan/`. It is built from d3c215d
(attempt-3 RTL 54249f7f plus the plan) and changes only what the plan's M1-M4
name. It adds no image parameter, so the final `image.tcl` keeps attempt 3's
parameters and only the source commit changes. Nothing here has been
implemented, loaded or timed on the board.

**Status (2026-10-20): deferred to R41; no R40 link.** The routed census of
attempt 3 ([`census/README.md`](census/README.md)) shows 12,799 endpoints
failing at 100 MHz behind the M1–M3 cone (WNS -0.919 ns), so this candidate
would link at about 91 MHz. The user chose to carry it into R41's image, built
together with the frequency work.

| ID | Change | Files |
|---|---|---|
| M1 | Registered host grant: core `c_ready` / NI mux depend on `host_busy` only; the host request goes out the cycle after the grant | `rtl/noc/tnca_idle_host_mux.v` |
| M2 | `host_allow` registered at its source (`tnca_top`) and again inside tile 0 (its SLR pblock); simulation assertion: no grant without live permission | `rtl/tnca_top.v`, `rtl/tile/tnca_tile.v` |
| M3 | Registered local copies of `memory_active_ports` (memory mode) and of aperture/limit (each AXI adapter, with the bound `min(size-64, 2^64-64-base)` precomputed); MMIO holds admissions for SETTLE=2 cycles after any change; simulation assertions on copy/register mismatch under traffic | `rtl/fpga/tnca_memory_mode.v`, `rtl/fpga/tnca_line_axi.v`, `rtl/fpga/tnca_multicore_mmio.v` |
| M4 | binary32 -> int32/uint32 per ISA sec 8.4: shift `{mantissa, 8'b0}` right by 158-cexp (never wraps), zero-shift tie guard, uint32 saturates at 2^32, negative uint32 -> 0 | `rtl/core/tnca_vector_fpu.v` |

## Verification (GB10, iverilog 12 / yosys 0.33, <= 4 cores, copy of this branch)

| Check | Result |
|---|---|
| `make lint` (68 files) plus an explicit elaboration of `rtl/board/tnca_multicore_hbm_xo.v` | pass |
| FPU oracle `tb_vector_fpu_f2i`: 181,632 vectors against `tools/model/f2i.py`, an exact-Fraction reading of the ISA text. Inputs are the SystemC set regenerated with the same seed (172,912 vectors; 17,015 differ on the old RTL) plus every boundary in exponents 120..160 | **0 differences**; attempt-3 RTL: 18,289 differences |
| New directed core program `t40_f2i`: large magnitudes, ties, ±2^31, 2^32, (-1, 1), inf, NaN; all 4 modes; lanes 0/15 + whole vector | pass in core, core-vrf, core-slowmem, core-ooo, core-r39sw5 (fast and ooo) |
| `make core core-vrf core-slowmem` (39 programs each, incl. decoder-fix set) | 117/117 pass |
| `make core-ooo core-r39sw5 unit` (incl. decoder 773, cvt_xls 866,495×2, fp32_fma, sfu, fpu_multi, asm round-trip) | pass |
| Host mux `tb_idle_host_mux`: directed; same-cycle probe that `c_ready` is independent of `host_allow`; 20,000 randomized cycles (1,837 grants, 1,837 host and 2,547 core responses) checking ownership, ready, grant condition and responses | pass; the attempt-3 mux fails the probe |
| `tools/r40/coherent_host_tests.py` (host mux + 3-core host grid, with the tnca_top grant assertion live) | pass |
| `tools/r40/memory_control_tests.py` (`tb_memory_mode`, `tb_mmio_memory_controls` ENABLE 0/1 with new settle checks, `tb_line_axi` limits 1/8, stages 0/2, host, legacy) | pass |
| Assertion sensitivity (scratch mutation probes, not committed): aperture changed under traffic; registered permission 1 with live 0 | both trip (line_axi assertion / tb_grid_cores concurrency check) |
| **Image-geometry simulation** `tools/r40/image_geometry_tests.py` on attempt 3's `image.tcl` (SETS 32, WAYS 8, all convergence options): code upload/readback, 2×455-program grids, memory controls 1/1, 1/8, 3/1, 3/8 each with a grid, coherent host + ownership handoff + partial-line handoff, deadline abort + quiesce | pass |
| Structural cone cuts `tools/r40/yosys_cone_cuts.py` (no Vivado) | pass for host mux, memory mode and AXI adapter; each check finds the chain in the attempt-3 RTL |
| System lanes: top, nuca, mesh, nuca-dco (+CI variants), TMU ranges; r40-grid-dispatch/multicore-shell/grid-deadline/core-parameters/memory-ports/line-axi/cache-maintenance/multicore-dma; multicore MMIO 1/1, 3/3, 3/3 board-top; native ports/features/pair; native four memory configurations; grid transport/runtime/backend; native smoke 3/8 cores; mmio-checks | all pass; see `regressions.md` |

## Vivado (FPGA host): not run

When this was checked, the FPGA host had 22 GB and later 16 GB available (`free -g`) and a
board measurement (archbench `op.matmul/hbm/fp16-m4096…`) was running.
The rule is at least 30 GB available and no host-latency experiment, so no
Vivado job was started.

`tools/r40/final_image_ooc.sh` holds the exact runs. It refuses to start below
30 GB. `OLD` is the attempt-3 source archive; `NEW` is this branch rsynced to
the host.

| Run | Command | Est. time / memory |
|---|---|---|
| Unit post-synthesis timing at 10 ns, old vs new: `tnca_vector_fpu`, `tnca_line_axi`, `tnca_memory_mode`, `tnca_multicore_mmio`, `tnca_idle_host_mux` | `tools/r40/final_image_ooc.sh units ~/tnca_src_r40_attempt03_54249f7 $NEW ~/tnca_reports_final_image/units` | 3-10 min and ≤ 6 GB each; 10 runs, ~1 h total |
| Kernel synthesis with the image generics | `tools/r40/final_image_ooc.sh kernel $NEW $NEW/evidence/r40/v2/convergence/attempt03/image.tcl ~/tnca_reports_final_image/kernel` | ~70 min, ~8.5 GB |
| Cross-SLR cone audit before (attempt-3 kernel DCP) / after, plus placement-hook dry-run | `tools/r40/final_image_ooc.sh audit ~/tnca_build/r40_attempt03_20261017/xo/tmp/link/vivado/vpl/prj/prj.runs/ulp_tnca_u50_1_0_synth_1/ulp_tnca_u50_1_0.dcp <image.tcl> $NEW ~/tnca_reports_final_image/audit_before` and the same with `.../kernel/synth.dcp` → `audit_after` | 10-20 min, 6-8 GB each |
| Census of the attempt-3 routed DCP (plan §2.3; decides C1) | `vivado -mode batch -source tools/r40/census_failing_endpoints.tcl -tclargs <level0_wrapper_routed.dcp> <out> 25000` | 50-70 min, 17-19 GB |

**Pass criteria:**
- Units: the new worst path is not worse than the old one. `tnca_line_axi`'s
  `req_ready` cone no longer starts at the aperture inputs.
- Audit-after: no `tile*` endpoint in the fanout of `memory_active_ports` or
  `home_link_idle`, and no `tile*/core/*` endpoint in the fanout of
  `host_allow`.
- Placement hook: `place_r40_multicore.tcl` resolves every hierarchy. M1-M4
  rename no placed instance; `g_host_mux.host_allow_q` is inside the tile.

## Plan deltas

- **M3 adds LUTs.** Precomputing the aperture bound as a minimum adds about one
  64-bit comparator and mux per adapter on the register side (off the request
  path), about +0.2k LUT in SLR0 in total. The plan said "LUT ≤ 0". The unit
  OOC will measure it.
- **What settle does.** A window access inside the settle is refused (status
  100) and a doorbell inside it is dropped, as for any busy device. The busy
  register reads 1 throughout. Real host accesses are far slower than 2 cycles,
  so software never sees it.
- **`tb_memory_mode` was out of contract.** It changed `single_port` and
  injected a response in the same cycle; it now waits one cycle.
- **`python/tnca/data/rtl` is not a tracked file.** It is a gitignored copy
  that `make r2-stage-data` regenerates from `rtl/`, so the plan's housekeeping
  item does not apply.
- **New Vivado-free structural evidence.** `yosys_cone_cuts.py` covers the
  module-level cuts now. The Vivado audit remains the chip-level proof.
