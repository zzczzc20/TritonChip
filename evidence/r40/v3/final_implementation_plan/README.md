# Plan for the last full U280 implementation (3 cores, 3 outlets, 100 MHz)

Status: **plan only**. No RTL was changed. No Vivado ran on the FPGA host and
nothing touched the card. The FPGA host was used read-only: reports and logs
were copied, and `ps`/`free` were run. The machine-readable form is
[`plan.json`](plan.json).

**Implementation status (2026-10-19, branch `r40-final-image`).**
- M1-M4 are implemented and simulation-verified. See
  [`../final_image_candidate/README.md`](../final_image_candidate/README.md).
- Pending: the Vivado runs (unit OOC, kernel synthesis, cross-SLR audit,
  census). They wait for at least 30 GB free on the FPGA host and no running
  board measurement.
- C1 is still decided by the launch rule in §2.5.

## 0. Summary

**What fails at 100 MHz.** Every reported failing path of attempt 3 (54249f7f,
image a88ebdf5) belongs to one combinational control cone that crosses SLRs
three times:

```
control/memory_active_ports (SLR1)
  -> memory_mode response mux / legal-source check
  -> outlet_rsp_valid (fo 517)
  -> home_link[2] idle (SLR1 -> SLR2 -> SLR1)
  -> all_idle -> host_allow (SLR1 -> SLR0)
  -> tile0 idle_host_mux use_host (fo 233)
  -> core c_ready -> LSU pending/partial, txf xbuf
```

The worst path is -1.002 ns over 14-16 LUT levels, and 92 % of its delay is
routing. The same path is the kernel's worst path at 90 MHz (+0.110 ns). It is
the only cone that phys_opt worked on: it re-placed 40 cells, all in tile0
`u_txf`. The existing reports print only 10 setup paths for the kernel clock,
which is 0.06 % of the 17,332 failing endpoints and 0.15 % of the
-6,679 ns TNS. **We do not yet know the rest of the failing population.**
§2.3 gives the read-only census that measures it, with its cost. Running it is
the first gate of this plan.

**Correctness.**
- No RTL or testbench file changed after 54249f7f. HEAD's hardware inputs are
  byte-identical to the attempt-3 archive: 103 of 105 bound inputs match, and
  the two that differ are diagnostic Tcl.
- The decoder fix 3d08eeb (illegal-opcode classification of killed
  instructions) is an ancestor of 54249f7f, and its `tnca_decoder.v` hash is
  bound in `attempt03/preflight.json`. **The loaded image contains it.** The
  physical-convergence note's warning applied to attempt 1 (cf330dc).
- Two hardware corrections are still outstanding: the vector-FPU float-to-int
  defects (§1.3) and the off-branch MSHR service snapshot 658c811 (§1.2).

**Recommended final candidate:** attempt 3 plus a short list of changes, with
the same flow, constraints, clock, cores, outlets, geometry and directive.

| ID | Change | Status |
|---|---|---|
| M1 | Registered host grant in `tnca_idle_host_mux`: core `c_ready` no longer depends on `host_allow` | must include |
| M2 | Registered `host_allow`, at the source (SLR1) and in tile0 (SLR0) | must include |
| M3 | Registered local copy of quasi-static memory controls, with an MMIO settle interlock. `memory_active_ports` goes to `tnca_memory_mode`; aperture and limit go to each `tnca_line_axi` | must include |
| M4 | Vector-FPU float-to-int conversion fixed to ISA §8.4, as a resource-neutral rewrite | recommended include |
| C1 | MSHR service snapshot 658c811 | include if the census or the slice OOC shows the slice MSHR->RAM path under 1 ns of margin |
| C2 | Registered home-link idle | only if the post-M1..M3 structural audit or the residual census still shows a two-SLR idle cone |
| C3 | Fixes for any residual census group | each needs its own OOC proof |

**Excluded:**
- the MMA change (R40-07.matrix): the ceiling is about 1.1x, short of the 1.2x
  gate, and it adds 20-30k LUT per core next to SLR0's 95 % CLB use;
- the 256 KiB code store: it closes no gate and moves `MEM_BASE`, which ripples
  through runtime, compiler and simulators;
- 32 B/core/cycle ingress: very high SLL and CLB risk, and R41-05 owns it;
- moving the TMU MMIO page.

**Process gaps to close before the link** (none is hardware):
1. **No test simulates the image geometry.** The benches use SETS 4/16 and
   leave the convergence parameters at 0. One 3-core board-top run with
   `image.tcl`'s parameters must be added (§5.1).
2. **The implementation watcher only diagnoses on exit 3.** On an
   auto-frequency-scaled exit it runs only the diagnose script. It must also
   run the image audit, the 100 MHz operating audit and the census (§5.3).
3. **The link takes about 11 h, not 4 h.** A3 ran 11 h 04 min end-to-end, and
   its route peaked at 30.9 GB on a 62 GB shared host. Other users' Vivado jobs
   currently hold about 40 GB, so the launch needs at least 40 GB free.

## 1. Correctness: hardware delta since the attempt-3 source

### 1.1 Committed after 54249f7f

| Scope | Commits | Effect on hardware |
|---|---|---|
| `rtl/`, `tb/` | none (`git diff 54249f7f..HEAD -- rtl tb` is empty) | none |
| `flow/fpga/` | 5 commits: audit/diagnose Tcl only (`audit_r40_operating_clock.tcl`, `audit_r40_placed_cores.tcl`, `diagnose_r40_checkpoint.tcl`) | report extraction only; not inputs of synthesis or placement |
| RTL generators | `rtl/gen/*` (XLS output) and `spec/r1_profiles.json` are byte-identical to the attempt-3 bound inputs | none |
| Software | many compiler/runtime commits, e.g. f9da3d5 (code-store overflow fallback) and 1a7c643 (TMU MMIO page excluded from allocation) | not hardware; independent of the image |

`python/tnca/data/rtl/` is a stale packaged copy (older `tnca_core.v`,
`tnca_decoder.v`, ...). It is not an input of the hardware build. It should be
refreshed or explicitly excluded before the final candidate is archived, so
that nobody mistakes it for the hardware source.

The check: `attempt03/preflight.json` lists 105 source-input hashes. Against
`HEAD` (b34daa6, re-checked after two new commits during this review; edd6221 gave the same result), 103 match byte for byte. The two that differ are
`flow/fpga/audit_r40_placed_cores.tcl` and
`flow/fpga/diagnose_r40_checkpoint.tcl`, both post-route diagnostics.

**Decoder fix.** 3d08eeb, "keep illegal opcode classification from poisoning
other contexts" (2026-10-16), is in `cf330dc..54249f7f`: attempt 1 lacks it,
attempts 2 and 3 contain it. The bound `rtl/core/tnca_decoder.v` hash equals
HEAD's. The loaded 90 MHz image therefore includes it, and the final image
inherits it. Its "native four configurations" regression must be rerun on the
final candidate (§5.1). That rerun re-verifies the fix on the new RTL; it does
not borrow a pass from another version.

### 1.2 Hardware work that exists but is not in the loaded image

| Item | Where | What it fixes | In loaded image | Final image |
|---|---|---|---|---|
| MSHR service snapshot | branch `r40-ms-service-snapshot`, 658c811, based on 54249f7f, `rtl/nuca/tnca_nuca_slice.v` +43/-14 | Captures the selected MSHR's set/way/bypass/tag on the existing E_SEL edge, so RAM addresses and fill controls no longer go through an 8:1 MSHR array mux. The predecessor module's worst path was 18 levels / 9.203 ns (7.757 ns net, address fanout 3,031) | no | **C1**, conditional. Cycle-equivalent (miter against 54249f7f plus an independent oracle, mutation detected); no physical screen yet. Include if the census shows slice data-RAM / MSHR families failing, or the slice OOC margin is under +1.0 ns |
| `stash@{1}` (r40-memory-ports batch) | `rtl/tnca_top.v` MEM_PORTS port list | superseded: the r40-memory-ports branch is merged and the multicore path uses `tnca_multicore_axi` | n/a | exclude |
| `stash@{0}` (packed-row optimisation) | compiler only | n/a | n/a | not hardware |

### 1.3 New correctness finding: vector-FPU float-to-int (ISA §8.4)

Source: `evidence/r40/v3/systemc_m1/README.md` §7. The oracle is
`tools/performance_harness/systemc/rtl_oracle/tb_vector_fpu_oracle.v`. 17,015
of 172,912 vectors differ from the ISA text. In
`rtl/core/tnca_vector_fpu.v:368-416`:

1. `csh = 8'd150 - cexp` wraps for |x| in [2^24, 2^31), so the right shift by
   249..255 gives 0 or ±1.
2. When `csh == 0` (odd |x| in [2^23, 2^24)), `chalf = 1 << (csh-1)` is 0, so
   RNE rounds up.
3. u32 saturates when `cexp >= 158`, so x in [2^31, 2^32) wrongly becomes
   0xffffffff.
4. u32 with x in (-1, 0) yields 0xffffffff under RNE (x < -0.5) and RMI.

No frozen suite uses these conversions (`systemc_m1/opcode_inventory.md`).
`t03_fp` checks only |x| ≤ 8 (`tb/core/progs/gen_t03_fp.py:172-193`), which is
why the defect escaped.

**Fix (M4).** Use one 32-bit right shifter of `{cman, 8'b0}` by
`sh = 158 - cexp`. For `cexp` in [127, 158], `sh` lies in [0, 31] and never
wraps. Use explicit `sh == 0` and `chalf == 0` guards for RNE ties. Saturate s32
at `cexp >= 158`, u32 at `cexp >= 159`, and every negative u32 to 0. This
replaces today's 64-bit shifter with a 32-bit one. Resources should be neutral
or slightly lower. Timing stays in the same stage (the `nf_v` -> `dly_v`
register), so the conversion depth does not grow; OOC must confirm this.

**Recommendation: include.**
- The final image is rebuilt anyway, and shipping a known silent wrong-answer
  path in a new image is worse than a small, exhaustively checkable edit.
- **Risk:** low. It is lane-local and combinational within an existing stage,
  with no state, protocol or SLR crossing.
- **Verification:**
  1. Rerun the 172,912-vector oracle against an **ISA-text** reference
     (`validate_numeric.py`), not against the RTL, and expect 0 differences.
  2. Add a directed core test covering [2^23, 2^32], ±2^31, the u32 negatives
     in (-1, 0), and all 4 rounding modes, in lanes 0 and 15.
  3. Run the existing `t03_fp`, `t21_cvt` and `tb_cvt_xls`.
  4. Run OOC of `tnca_vector_fpu` at 10 ns: the worst path must not get worse.
- **Software follow-up:** the SystemC model "follows the RTL", so it must
  switch to the fixed behaviour for the new image.
- **If the OOC result regresses:** exclude M4 and record the defect as a known
  erratum. It must not block the timing fixes.

### 1.4 Not recommended as hardware changes

- **TMU MMIO page inside the device window** (`TMU_BASE=0x7fff0000`,
  `rtl/nuca/tnca_nuca_slice.v:90,861`). The software reservation 1a7c643 is
  audited. Moving the page changes `compiler/python/tnca_cc/dco.py` and the
  runtime address map. **Exclude.**
- **`python/tnca/data/rtl` drift.** Not hardware; housekeeping (see §1.1).

## 2. 100 MHz: failing path groups and fixes

### 2.1 What the existing reports contain

All of the following were read-only copies from
`~/tnca_build/r40_attempt03_20261017/.../impl_1/`, preserved in
`attempt03_impl1_reports.tgz` and `attempt03_impl1_runme.log.gz`.

| Source | Command | Kernel setup paths printed |
|---|---|---|
| `hw_bb_locked_timing_summary_routed.rpt` (sha be756df0…, 100 MHz) | `report_timing_summary -max_paths 10` | 10 |
| `dr_timing_summary.rpt` (100 MHz) | `report_timing_summary -slack_lesser_than 0` (max_paths defaults to 1) | 1 |
| `qualified90_raw.tar.gz` `timing.rpt` / `setup_paths.rpt` (90 MHz what-if) | summary (1 per group) / `report_timing -max_paths 25` (whole design: the 25 are platform paths) | 1 (kernel WNS +0.110 ns: the same path) |
| `runme.log` | placer / phys_opt / router progress | phys_opt critical-path phase processed 40 nets, **all** in `g_tile[0].u_tile/u_core/u_txf` (`xbuf`, `t_i3`) |

**Design totals (100 MHz, `clk_kernel_00_unbuffered_net`).**
- WNS -1.002 ns, TNS -6,679.026 ns.
- 17,332 failing of 1,056,119 endpoints.
- Hold: WHS 0.000 ns, 0 failing. Pulse width: 0 failing.
- Mean violation per failing endpoint: -0.385 ns.

How the violation evolved through implementation (`runme.log`):

| Stage | WNS (ns) | TNS (ns) |
|---|---|---|
| post-place estimate (placer post-commit) | -1.154 | -2,861 |
| after `phys_opt_design` (default directive) | -1.129 | -2,849 |
| router initial estimate, partial routes | -0.559 | -544 |
| router rip-up, global iteration 0 | -2.230 | -12,545 |
| end of route | -1.006 | -7,114 |
| after the router's internal post-route physical optimisation | -1.002 | -6,679 |

**About 57 % of the final TNS appeared during routing.** That is congestion
detour, consistent with routing estimated congestion level 6 (64x64), worst in
SLR0, and 2,061 CLBs with high pin utilisation.

### 2.2 Groups visible in the reports

`attempt03_100mhz_reported_path_groups.json` was produced by
`tools/r40/group_failing_paths.py --report …`:

| Group (source family -> destination family) | Reported endpoints | Worst slack | Levels | Logic / route (ns) | Data-path SLR hops | High-fanout nets on path |
|---|---|---|---|---|---|---|
| `control/memory_active_ports_reg` -> `g_tile[0]…/g_lsu_pair.u_lsu/g_unit[1].u/pending_reg` | 2 | -1.002 | 14 | 0.872 / 10.311 | 1->2, 2->1, 1->0 | `outlet_rsp_valid` fo 517 (0.770); `host_busy`/`use_host` fo 233 (0.282); outlet `rd` fo 86 |
| same source -> `…/g_unit[1].u/partial_reg` | 1 | -0.994 | 14 | 0.955 / 10.220 | same | same |
| same source -> `g_tile[0]…/u_core/u_txf/xbuf_reg` (R pin) | 7 | -0.974 | 16 | 0.869 / 9.915 | same | same; `u_txf/mem_reg…` fo 233 |

The cone, step by step:
- `memory_active_ports_reg[1]` (FDSE at SLICE_X101Y304, SLR1) feeds the
  `tnca_memory_mode` `legal_source`/`single_port` mux (`rtl/fpga/tnca_memory_mode.v:47-60`).
- That drives `s_rsp_valid` into the home links (fo 517; this also gates the
  512-bit reply capture) and into `tnca_line_link.idle`, which includes the
  incoming `m_rsp_valid` combinationally (`rtl/common/tnca_line_link.v:34`).
- `home_link_idle[2]`: stage 0 sits in SLR2 and stage 1 in SLR1 (pblocks), so
  the OR tree goes out to SLR2 and back. Net 2.404 ns.
- `all_idle` -> `host_allow` (`rtl/tnca_top.v:219,226,257`), placed in SLR1,
  crosses into SLR0 (1.075 ns).
- `tnca_idle_host_mux` (`rtl/noc/tnca_idle_host_mux.v:21-25`) computes
  `offer_host = host_allow && !c_valid && h_valid` and
  `c_ready = n_ready && !(host_busy || offer_host)`.
- From there the core's memory handshake (LSU `mem_req_ready`, txf
  `tm_req_ready`) runs into `pending`, `partial`, the `xbuf` resets and so on.

Only tile 0 has this path: `host_valid && t==0` (`rtl/tnca_top.v:488`) ties
`h_valid` to 0 for tiles 1 and 2.

**Assessment (inference, not measurement).** The endpoints downstream of
tile0's `c_ready` (LSU lane state, txf 1024-bit `xbuf`, `t_i3`, …) number in
the thousands. A post-place phys_opt that touched only this cone, plus a
-0.385 ns mean violation, are both consistent with this cone being a large
share of the 17,332 endpoints. -nworst 1 per endpoint would also hide every
other failing path of those endpoints. The census below settles it.

### 2.3 The census that covers TNS -6,679 ns

**Script:** `tools/r40/census_failing_endpoints.tcl`. It is read-only: it
opens the checkpoint and never writes one. Run it on the A3 routed DCP
(`level0_wrapper_routed.dcp`, 649 MB):

```
vivado -mode batch -nojournal -log census.log \
  -source tools/r40/census_failing_endpoints.tcl \
  -tclargs ~/tnca_build/r40_attempt03_20261017/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp \
           ~/tnca_reports_r40_final_plan/census_a3 25000
python3 -I tools/r40/group_failing_paths.py --tsv .../failing_endpoints.tsv \
  --tns -6679.026 --endpoints 17332 --out census_groups.json
python3 -I tools/r40/group_failing_paths.py --tsv .../residual_failing_endpoints.tsv --out residual_groups.json
```

What it does:
1. Runs `get_timing_paths -setup -group clk_kernel_00_unbuffered_net -max_paths 25000 -nworst 1 -unique_pins -slack_lesser_than 0`.
   It writes one TSV row per failing endpoint: slack, start/end pin, levels,
   data/logic/net delay, skew, start/end SLR. The enumerated TNS must equal
   -6,679.026 ns.
2. Runs `report_design_analysis -timing -setup -max_paths 25000 -show_all`,
   plus `-csv` where supported, for SLR crossings, high fanout and primitives.
   It also writes the logic-level distribution and full detail of the 200 worst
   paths.
3. Runs a **residual what-if.** In memory only, and never saved, it excludes
   paths from `control/memory_active_ports_reg*` and through `*host_allow*`,
   then repeats the census. This exposes the failing paths that -nworst 1 hides
   behind the cone, which is the population that would remain after M1-M3,
   measured on real A3 routes. It is not a constraint and qualifies nothing.

**Cost**, measured on the same DCP by the 90 MHz audit (`runner.log`):

| Step | Time | Peak memory |
|---|---|---|
| `open_checkpoint` | 10.5-12.4 min wall, ~15.5 CPU-min | ~13.6 GB |
| first `update_timing` | ~10.6 min wall, ~17 CPU-min | ~16.2 GB |
| `report_timing_summary` (reference) | 2 min | |
| `report_drc` (reference) | 4 min | |
| census queries (est.) | 10-20 min | +1-2 GB |
| what-if `update_timing` + residual census (est.) | ~15-25 min | |

- **Total:** about 50-70 min wall, about 1.5 CPU-hours at
  `general.maxThreads 2`, peak about 17-19 GB RSS.
- **Not now.** The FPGA host has 62 GB. At the time of reading, other users'
  Vivado jobs held about 40 GB RSS (28.5 GB + 12 GB) and only about 19 GB was
  available, while the board campaign runs.
- **When:** after the board measurements finish and when at least 24 GB is
  free.

### 2.4 Fixes for the known cone (no false paths, no multicycle)

The owner's rule is that nothing is excluded by constraint. Every fix below
cuts the cone with a register and proves equivalence through protocol
invariants, assertions and simulation.

**M1: registered host grant**
- **File:** `rtl/noc/tnca_idle_host_mux.v`.
- **Change:**
  - Grant cycle: when `host_allow_q && !c_valid && h_valid && !host_busy`, set
    `host_busy <= 1`. Nothing is issued in this cycle.
  - Issue cycle: `n_valid = host_busy ? (!host_issued && h_valid) : c_valid`,
    and `c_ready = n_ready && !host_busy`.
  - `use_host` becomes the register `host_busy`.
- **Why it is safe:** a grant only happens when `!c_valid`, and `c_ready`
  becomes 0 the next cycle, so no core request is lost. The host response
  matching (`HOST_TAG`) is unchanged.
- **Cost:** host word accesses are one cycle longer. These are MMIO-window
  accesses, so performance is unaffected.
- **What it removes:** `host_allow` drops out of tile0's `c_ready`, the LSU
  and txf. This cuts the fo-233 net and all 10 reported endpoints.
- **What stays:** `h_valid` still reaches tile0 from SLR1 shell logic, as
  `host_valid && !code_selected && … && (!cmd_valid || host_busy)` at
  `rtl/tnca_multicore_shell.v:194`. After M1 it feeds only the grant flop and
  the host-side `n_valid`, never `c_ready`.
- **Tests:** the existing `tb/tb_idle_host_mux.v` checks stay valid. Add two:
  - a structural check that `c_ready` cannot change in the cycle `allow` rises
    (`#1` probe);
  - a randomized interleaving of core and host traffic.

**M2: registered `host_allow`**
- **File:** `rtl/tnca_top.v:257`.
- **Change:** register `host_allow` once at the source, beside dispatch in
  SLR1. Register it again in tile0's host mux, so the copy sits in the SLR0
  pblock.
- **Why it is safe (invariant):** `host_allow_q` is consumed only while
  `h_valid`. While `h_valid`, MMIO is in `host_state==ISSUE` with
  `state==IDLE`, so no command, maintenance or code admission can happen.
  `all_idle` was required at admission
  (`rtl/fpga/tnca_multicore_mmio.v:123-125`; doorbell `:130`, memory controls `:111`). Every input of `host_allow` is
  therefore constant until the host access itself starts. A stale 1 after the
  access is harmless because `h_valid` is 0. A stale 0 only delays the grant.
- **Assertion (simulation):** a grant with the live `host_allow == 0` is fatal.
- **Removes:** the combinational SLR1 -> SLR0 hop.

**M3: quasi-static memory controls as registered local copies, plus an MMIO
settle interlock**
- **MMIO** (`rtl/fpga/tnca_multicore_mmio.v`): a write to 0x110 (active
  ports / outstanding) or to `axi_base`/`axi_size` already requires
  `state==IDLE && host_state==IDLE && all_idle`. Add a 2-bit `settle` counter
  that holds off every admission (doorbell 0x48, window 0x88, and the code/
  maintenance paths that go through them) for 2 cycles after such a write. The
  busy register 0x48 reads busy during settle.
- **Memory mode** (`rtl/fpga/tnca_memory_mode.v`): `single_port` becomes a
  registered copy of `memory_active_ports==1`. The response mux and
  `legal_source` then depend only on outlet-link reply registers and a local
  flop.
- **AXI adapter** (`rtl/fpga/tnca_line_axi.v`): registered local copies of
  `axi_base`, `axi_size`, the precomputed `axi_size-64` limit and
  `admission_limit` (the 64-bit compare chain in `legal_address` and
  `admission_ok`, `:61-65`). These sit in the adapter's SLR0 pblock, so the
  MMIO (SLR1) -> SLR0 crossing becomes a register-to-register hop.
- **Proof obligation:**
  - Settle is longer than the copy latency, and every request-producing path
    starts with an MMIO admission.
  - So no request or response can see a copy that differs from the register.
  - Assertion: any request or response at `tnca_memory_mode` or
    `tnca_line_axi` while a copy differs from its register is fatal.
- **Cost:** about 3 x (2 x 64 + 8 + 64) + 2 FFs, roughly 600 FFs, almost all
  in SLR0 adapters. LUTs are roughly unchanged; precomputing `axi_size-64` into
  the copy also removes one 64-bit subtractor from the request-ready path.

**C2: registered home-link idle (conditional)**
- After M3, `home_link_idle` is fed only by local registers and the outlet
  reply register (via `legal_source`).
- Its remaining hop is SLR2 stage-0 occupancy -> SLR1 OR, then to MMIO and
  handoff consumers: one SLL plus a few LUTs, roughly 4-6 ns.
- If the structural audit or the residual census still shows it failing:
  - export per-stage registered occupancy from each stage's own SLR;
  - drop the redundant `m_rsp_valid` term. Every in-flight response is held in
    a register of the outlet link (counted by `link_idle`/`memory_drained`),
    or its MSHR keeps the slice non-idle.
  - Proof needed: exhaustive drain tests show `all_idle` is never true while a
    response is in flight, under random traffic with writes and maintenance.

**Expected effect on this cone** (from the A3 path's own increments, not
measured): the longest remaining segment is about 4-6 ns, against a 10 ns
period:
- MMIO/handoff consumers of `all_idle`: one SLL hop;
- `host_allow` -> tile0 FF: about 1.1 ns net;
- `host_busy` -> LSU `pending`: about 2.6 ns.

**OOC experiments that prove the cone is cut**, run with
`tools/r40/audit_cross_slr_cones.tcl` on the kernel synthesis DCP. Do it once on
A3's `ulp_tnca_u50_1_0.dcp` as the "before", and once on the candidate's
`flow/fpga/synth_r40_multicore.tcl` output as the "after". Use
`place_r40_multicore.tcl` as the hook so every endpoint is labelled with its
SLR pblock. Pass criteria:

1. `fanout_memory_active_ports.tsv` has no `tile*` endpoint. It ends at MMIO
   and the memory-mode/adapter copies.
2. `fanout_host_allow.tsv` has no `tile*/core/*` endpoint. Only the host-mux
   registers remain.
3. `fanout_home_link_idle.tsv` has no `tile*` endpoint.
4. The longest path through these nets has at most 6 LUT levels and starts and
   ends at registers in at most two adjacent SLR pblocks.
5. The placement hook runs without error on the new hierarchy. This dry-run
   prevents the about-1.2-hour-into-link failure mode.

### 2.5 Residual groups (after the census)

Each group in `residual_groups.json` worse than 0 ns must be classified before
the RTL freeze. Use the rules below; the evidence pointers are what we already
know.

| Likely class | Evidence today | Fix pattern | OOC proof |
|---|---|---|---|
| Slice MSHR -> data RAM (8:1 MSHR mux into 3,031-fanout address) | R4 module routed: 18 levels / 9.203 ns (7.757 net) at 10 ns, module WNS +0.361 | C1 (658c811) | `synth_r40_slice.tcl`/module flow at the A3 geometry: worst path improves, RAM primitives unchanged, WNS ≥ +1.0 ns |
| High-fanout per-core enables / resets that the placer moved to BUFG: LSU `ln_wlo` (4,080), `lane_addr` (≈1,700-1,760), VRF `d_valid` (1,653), dispatch queue reset/instret (4,723-4,728), TMU `tb` clear (1,081), txf `t_rank` (1,014) | `runme.log` BUFG insertion list (25 inserted, 3 skipped for timing) | Register-replicate the driver per region (`MAX_FANOUT` on a pipelined copy, or an explicit tree of 2-3 registered copies). Not `DONT_TOUCH` | tile OOC: no path through these nets in the top 1,000 |
| Paths in SLR0's congested region (`CLEM_X27..66 Y77..143`, initial congestion 64x64) | router congestion report | Reduce SLR0 pressure: M1/M3 add nothing to SLR0's CLB count; do not add SLR0 logic (M4 is neutral) | census-residual WNS ≥ 0 on the A3 routes, plus tile OOC |
| Anything else | census | Pipeline, or cut with an explicit invariant | unit OOC with WNS ≥ +1.0 ns |

**Rule for the launch decision.** Proceed to the final link only if both of
the following hold:
1. Either the residual what-if census on A3 routes has WNS ≥ 0, or every
   residual group has a committed fix whose unit OOC shows WNS ≥ +1.0 ns (10 %
   of the period).
2. The structural audit passes.

The +1.0 ns margin comes from A3 itself: route-stage TNS growth was 57 %, and
OOC module slack (R4 +0.361 ns) does not carry over into a congested full
chip. **If the census shows a large residual population with no fix, report it
and do not spend the attempt.** This is P0/P1 directive 11.

## 3. Routing risk

What attempts 1 and 2 hit, and what A3 already contains:

| Countermeasure | A1 cf330dc | A2 0c39793 | A3 54249f7f |
|---|---|---|---|
| `SSI_SpreadSLLs` place directive; default phys_opt/route | yes | yes | yes |
| Whole-SLR pblocks: tiles, private code RAM, mesh routers, split link stages, outlets -> SLR0, ownership -> SLR1 | yes | yes | yes |
| `USER_SLL_REG` on response payload FF pairs (home links 0/2, three outlets) | no | no | **yes** (515/517 pairs each) |
| Local TMU replica per SLR (`DCO_LOCAL_REPLICAS=1`) removing the TMU output broadcast (1,439 three-SLR nets in A2) | no | no | **yes** |
| Registered TMU writes, TLW pipe ×2, registered victim select/rank, cache data pipeline | no | no | **yes** |
| Balanced victim tree, synchronous tag bank, `KEEP_HIERARCHY` on slice | no | yes | yes |
| Registered FIFO heads (p1_head_fifo_screen) | no | no | **no**: the screen itself decided against it (RAM32M16 42->0, FF 5->1,166, LUT 7->587 on a 582-bit FIFO) |

The outcome:

| Run | CLB % SLR0/1/2 | Peak SLL column (SLR1-2 / SLR0-1) | SLL aggregate SLR0↔1 / SLR1↔2 | Result |
|---|---|---|---|---|
| 09ebe06 | 97.59 / 96.54 / 86.54 | 2,940 (204 %) / 1,778 (123 %) | | congestion level 7 |
| A1 | 95.83 / 90.81 / 87.37 | 2,926 (203 %) / 1,619 (112 %) | | route failed, congestion level 7 |
| A2 | 93.53 / 87.70 / 86.07 | 3,062 (213 %) / 1,787 (124 %) | 60.9 % / 40.4 % | 2 resource-conflict + 2 antenna nets |
| **A3** | **95.24 / 88.39 / 85.41** | **no Route 35-3311 "high localized SLL demand" warning** | 62.47 % / 35.06 % (2,580 Laguna-registered) | fully routed, 0 overlaps |

So A3 removed the localized SLL-column overload. A2's router printed the
per-column table with 213 %/124 %; A3's router printed no such table. A3 still
has two weak points:
- **SLR0 CLB at 95.24 %.** A1 failed at 95.83 %, so the headroom is about
  0.6 points, roughly 330 CLBs or 2.6k LUT.
- **Initial routing congestion level 6.**

Hold slack is 0.000 at both 100 and 90 MHz, which means the router fixed hold
with detours.

Effect of the recommended changes:

| Change | SLL | SLR0 CLB | Notes |
|---|---|---|---|
| M1 | 0 | +few FF | inside tile0 |
| M2 | 0: the same 1-bit crossing becomes registered | +1 FF | |
| M3 | 0: the existing 2×64+8+64-bit MMIO->adapter crossings become reg->reg | ~+600 FF, LUT ≤ 0 | FFs pack into existing CLBs; FF use in SLR0 is 18 % |
| M4 | 0 | ≤ 0: 32-bit instead of 64-bit shifter per lane | all three cores |
| C1 | 0 | ~+60 FF per slice, fewer mux LUTs | |
| MMA x2 rows (excluded) | 0 | +20-30k LUT per core: SLR0 to ~98 % CLB | A1 failed at 95.8 % |
| Code store 256 KiB (excluded) | 0 | +32 RAMB36 per core (≤ 26 %); address fanout | needs `MEM_BASE` move |
| 32 B ingress (excluded) | ≈ +860 per boundary | +10-20k LUT per SLR | the failing mechanism of A1/A2 |

Residual routing risks, and how to handle them:
1. Placer randomness from netlist perturbation. Keep the change set minimal and
   do not touch pblocks or directives.
2. **Placement hook brittleness.** A renamed hierarchy is fatal about 1.2 h in.
   The structural audit dry-runs the hook on the new synthesis DCP.
3. Do not add `DONT_TOUCH`/`KEEP` beyond A3's; the P1 directive applies.
4. **Host memory.** A3's route peaked at 30.9 GB, with free memory falling to
   421 MB. Launch only with at least 40 GB free, and keep the census and OOC
   jobs off the host during the link.

## 4. Optional hardware features

| Feature | Gate | Benefit | Resource / timing / routing risk | Verification cost | Recommendation |
|---|---|---|---|---|---|
| (a) MMA: overlap B prefetch / A load | R40-07.matrix: S_geo ≥ 1.2 (v3: deferred, target only) | FP16 K16 MMA is 311 cycles (16 A + 32 B + 256 MAC + 7); overlap brings it to about 265-280, so matmul ≤ 1.07x, linear ≤ 1.05x, S_geo ≈ 1.02-1.05. 3 of the 14 subset points are fp32 and cannot use MMA | low (+300 FF, +0.5-1k LUT per core) | new mode bit-equivalence tests, `TNCA_RTL_MMA` MMIO consumer, 14-point device sim, same-image A/B | **exclude**: cannot reach 1.2x |
| (a') MMA: 2 rows per cycle (32 MAC/cycle) + prefetch | same | S_geo ceiling about 1.10-1.13; an infinitely fast MMA caps it near 1.3 | +20-30k LUT, +4-6k FF, +32-48 DSP per core; SLR0 to ~98 % CLB; the FMA operand mux was the historical OOC worst path (`doc/r39_report.md`) | high | **exclude**: fails 1.2x, and is the A1 failure regime. Belongs to R41-04 (S_geo ≥ 1.5) |
| (b) Code store 256 KiB (`CODE_WORDS` 16384->32768) | none. The refused kernels have fallbacks: GEMM tile 16,64,16 (136,256 B) falls back under the frozen compile rule; int64 transposed pointwise (250,880 B) falls back to 16 lanes in f9da3d5 | about nil: int64 would still exceed the 240 KiB wide-code limit; the uncalibrated SystemC model predicts 16,64,16 is slower | +32 RAMB36 per core (BRAM ≤ 26 % per SLR); low timing risk. **But `MEM_BASE` 0x20000 must move to ≥ 0x40000** (`tnca_multicore_mmio.v:92` guard), which changes every device data address: runtime heap base, inductor/compiler limits, `tb/device`, archbench platform | B1 conformance, upload/readback across the old boundary, `bad_pc`, LRU eviction, full native77/HBM reruns under a new memory map | **exclude** |
| (c) 32 B/core/cycle ingress | v3: R40-04.ingress deferred; R41-05 owns it | the measured core ceiling is 2.3-3.7 B/cycle even with ideal RAM (`doc/r40_hbm.md`): the port is not the limit | NoC flit ~431/261 bits, wider routers/NI/slice port: ≈ +860 SLLs per boundary, +10-20k LUT per SLR | nuca, nuca-dco, mesh, top, e2e, R2 fault/drain, coherence handoff, misaligned transfer | **exclude**: very high risk on the exact A1/A2 failure mechanism |

Contract note: v3 binds every R40 gate to image a88ebdf5 and says "R40 内不再做实现或烧录".
A new image can only serve deferred items or R41 unless the owner amends the
identity. See open question 1.

## 5. Verification checklist before link

### 5.1 RTL (GB10, iverilog / yosys 0.33, ≤ 4 cores)

The rules come from `doc/r40_tasks_and_acceptance.md` §3.4:
- l.146 inherits the R39 merge gates: `make r37`,
  core/core-vrf/core-slowmem/core-ooo/top/nuca, `r38-baseline-check`,
  `r38-profiler`, `r38-torch`, `r38-tdesc`, and the applicable R39 role/pair
  regressions.
- l.150 orders the hardware steps: lint/elaboration, then directed
  fast/slow/out-of-order/backpressure tests, then OOC, then link. It also
  requires reporting per-SLR and whole-chip LUT/FF/BRAM/URAM/DSP, SLR
  crossings, post-route setup/hold, CDC and DRC, and forbids signing off a
  silently down-clocked image.
- l.100 covers 1/2/4 outlets with fast/slow/out-of-order responses.
- l.474 covers 1/2/3/4-core regressions and an 8-core smoke test.

Run everything on the exact candidate tree, and bind every result to the
changed-file hashes.

| Step | Command / target | Covers |
|---|---|---|
| Lint | `make lint` (iverilog -g2005 -t null over rtl/{common,core,gen,nuca,noc,tile,fpga} + rtl/*.v), **plus an explicit elaboration of `rtl/board/tnca_multicore_hbm_xo.v`**, which `make lint` does not cover | all |
| Core | `make core core-vrf core-slowmem core-ooo core-r39sw5` (the image profile core-vrf-sync-v5: banked VRF, RCTX_SWITCH=2, LSU_INFLIGHT=2, RPAIR, MTX_ABUF, …; `core-r39sw5` is **not in CI**) and the decoder-fix set `make -j3 core core-vrf core-slowmem` + standalone `tb_decoder` | M4, decoder fix |
| M4 numerics | `tb_vector_fpu_oracle` (172,912 vectors) vs the ISA-text reference: 0 diffs. New directed core test: abs(x) in [2^23, 2^32), ±2^31, u32 negatives in (-1, 0), 4 modes, lanes 0/15. Plus `t03_fp`, `t21_cvt`, `tb_cvt_xls` | M4 |
| Host mux | `tools/r40/coherent_host_tests.py` (`tb_idle_host_mux` + host grid), extended with the `#1` probe that `c_ready` is independent of `allow`, randomized core/host interleaving, and the stale-grant assertion. Also `make r40-coherent-host` | M1, M2 |
| Memory controls | `tools/r40/memory_control_tests.py` (`tb_memory_mode`, `tb_mmio_memory_controls` ENABLE=0/1, `tb_line_axi`), `tools/r40/line_axi_tests.py`, `make r40-memory-ports r40-line-axi r40-cache-maintenance r40-multicore-dma`, plus a new settle-window test (a doorbell or window access immediately after a 0x110/aperture write must be held or refused) and copy-mismatch assertions | M3 |
| Native four configurations | `python3 tools/r40/memory_control_tests.py --native --python ~/tc_env/torchenv/bin/python --out <fresh>`: resident Torch graph at outlets/outstanding 3/8, 1/8, 1/1, 3/1 (≤ 3600 s; CI runs it **without** `--native`). This is the regression that exposed the decoder reservation-X | M1-M4 |
| NUCA / TMU | `make nuca nuca-dco` and the CI variant `make nuca-dco DCO_POLICIES='2 74' DCO_SEEDS='1 7' IVFLAGS='-g2012 -I rtl/common -Ptb_nuca_slice.DCO_REGISTERED_RANGES=1 -Ptb_nuca_slice.REGISTER_VICTIM_OWNER=1 -Ptb_nuca_slice.EPOCH_CHURN=1'`, `tb_tmu_registered_ranges` with PORTS 3 and 8, and 658c811's miter rebased (`tb/nuca/service_snapshot.v`, oracle, mutation) | C1 |
| System | `make mesh top`, `make r40-core-parameters r40-multicore-shell r40-grid-deadline r40-grid-dispatch`, `make r40-multicore-mmio-checks`, `tools/r40/multicore_mmio_tests.py --cores 1 --ports 1` / `--cores 3 --ports 3` / `--cores 3 --ports 3 --pipeline 2 --board-top`, `native_memory_ports_tests.py`, `native_multicore_features.py [--pair] --backend fpga-multicore-sim --cores 3 --pipeline 2 --board-top`, `make r40-physical-grid-transport r40-multicore-native-smoke r40-physical-chunks r40-pipeline-checks` | all |
| **Image-geometry simulation (new, required)** | The standard benches do **not** use the image parameters. `tb_nuca_slice` defaults to SETS=16/WAYS=4; the board TB `tb/fpga/tnca_multicore_board_regs.v` uses SETS=4/WAYS=2 with `DCO_LOCAL_REPLICAS`, `NUCA_REGISTER_VICTIM_*`, `NUCA_CACHE_DATA_PIPELINE`, `DCO_TLW_PIPE_STAGES` and `DCO_REGISTER_WRITES` at 0. Add one 3-core board-top run with exactly `image.tcl`'s parameters (SETS=32, WAYS=8, those parameters at 1/2) on the coherent-host, memory-control and cache-maintenance scenarios | all |

Cost: the split CI took 1 h 47 min wall with lanes in parallel (`attempt03/split_ci_terminal_success_20261018.json`). The R40 lanes alone sum to about 8 h serial, so expect about 2.5-3 h on 4 GB10 cores for the R40 set. The core/NUCA set adds about 1 h. Exact-source GitHub CI (`.github/workflows/ci.yml`: check, triton, tilebench, torch, 5 r40-grid lanes, 7 r40-mmio lanes) must also pass on the frozen commit.

### 5.2 OOC on the FPGA host (only when the board is idle)

1. **Census** of the A3 DCP (§2.3): before any RTL freeze.
2. **Kernel OOC synthesis** of the candidate: `flow/fpga/synth_r40_multicore.tcl
   ROOT image.tcl OUT`, about 1 h 10 min and 8.5 GB. Then:
   - `flow/fpga/audit_r40_multicore_placement.tcl DCP image.tcl REPORT`, the
     existing pblock-contract dry-run of the placement hook;
   - `tools/r40/audit_cross_slr_cones.tcl`, once on A3's
     `ulp_tnca_u50_1_0.dcp` (before) and once on the new synthesis DCP
     (after), checked against the §2.4 criteria;
   - `report_utilization -hierarchical` deltas per tile (expect ≤ +0.1 %
     CLB-equivalent in SLR0).
3. **Unit synthesis estimates** use `flow/synth/vivado_ooc.sh <unit> 10 [PARAM=…]`,
   about 5 min per unit; routed results have run about 1.6x the synthesis
   estimate. **Unit place and route** reuses the module P&R script
   `screen_local_tmu_rank.tcl`, which lives in
   `attempt03/r40_hit_owner_module_20261017.tar.gz` and not in `flow/fpga`:
   3 slices + 3 TMUs, SLR pblocks, `SSI_SpreadSLLs`, ≤ about 51 min, last result
   WNS +0.361 / WHS +0.010. Promote that script into `flow/fpga` first. Then
   run, at 10 ns and the A3 geometry:
   - `tnca_vector_fpu` (M4): the worst path must not regress.
   - slice + TMU module (C1, via the R4 module flow): WNS ≥ +1.0 ns, RAM
     primitive mapping unchanged (96 RAM32M16 tag roots, data RAM type).
   - `tnca_line_axi` (M3): `legal_address` path WNS ≥ +1.0 ns.
   - `tnca_idle_host_mux` + LSU pair (M1): no path from `host_allow`.
4. **Per residual census group:** its unit OOC (§2.5).

**What "proven to change the failure mechanism" means here:**

| Change | Proof |
|---|---|
| M1/M2/M3 | the structural audit shows the A3 cone's connectivity gone (criteria 1-4), and the residual census shows what remains |
| M4 | correctness only. Proof that it does not *add* a mechanism: no FPU path in the core OOC top 100 that was absent before |
| C1 | the slice OOC worst path leaves the MSHR mux -> RAM address family, or gains ≥ 1 ns |

An area reduction alone is never accepted as proof.

### 5.3 Build and link

The flow is identical to A3:
- `flow/fpga/run_r40_attempt01.sh FRESH_ROOT image.tcl`. It sources Vivado/Vitis
  2023.2 + XRT and sets `TNCA_IMAGE_CONFIG`, `VPP_JOBS=2` and
  `LINK_EXTRA=--vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=SSI_SpreadSLLs`.
- `package_xo.tcl` uses `ooc.xdc` with `USED_IN {synthesis out_of_context}`.
- `link.sh` runs `v++ -l --config fpga/u280_multicore_hbm.cfg` (three masters,
  each `HBM[0:31]`), with `PLACE_DESIGN.TCL.PRE=place_r40_multicore.tcl` and
  `--kernel_frequency 0:100`.
- `image.tcl` is regenerated by
  `tools/r40/image_profile.py --profile core-vrf-sync-v5 --multicore-hbm`,
  with the A3 parameters plus any new one.
- There is no new XDC exception.

`link.sh` fails closed:
- exit 3 means the tool auto-scaled the clock;
- exits 4/5 mean the xclbin readback is unreadable or not 100 MHz;
- `LINK_RECHECK=1` re-checks without relinking.

Sequence:
1. Freeze the candidate commit.
2. Pass exact-source CI.
3. Make the immutable archive: `git archive` snapshot plus the 3 config files.
   Unpack it on the host as `~/tnca_src_r40_attempt04_<sha>`. The preflight
   hashes every input, as `attempt03/preflight.json` did for 105 inputs.
4. Run preconditions modelled on A3's `launch_once.py`. A3 checked its
   predecessor module P&R; the final candidate checks its §5.2 screens:
   - every unit/module P&R `runner.exit` == 0 with WNS/WHS ≥ 0 at 10.000 ns
     (≥ +1.0 ns for changed units) and 0 route errors;
   - the structural audit and hook dry-run pass;
   - per-SLR TMU state and tag-RAM mapping unchanged;
   - all input hashes match the archive;
   - the contract check passes;
   - a fresh build directory exists;
   - an exclusive reservation is recorded in `implementation_budget.json`.
5. Launch detached, with `tools/r40/watch_implementation.py`.
6. **Fix the observer gap first.** On any non-zero exit, including exit 3 as in
   A3, the watcher only runs `diagnose_r40_checkpoint.tcl`. It must also run
   `audit_r40_image.tcl` (route status, CDC, clock interaction, SLR
   utilisation/SLL matrix, min/max timing, bus skew; no DRC),
   `audit_r40_operating_clock.tcl DCP OUT 100 placement` (timing with
   unconstrained paths, 25 setup/hold paths, CDC details, check_timing, DRC,
   per-core SLR), and `census_failing_endpoints.tcl` when WNS < 0.

**Timeline**, measured on A3 (`runme.log`, `build/`): link started 21:20:46Z;
v++ rc=0 at 08:24:57Z. That is **11 h 04 min end-to-end, not 4 h.**

| Stage | Time |
|---|---|
| packaging + IP/kernel synthesis (kernel `synth_design` 1 h 08 min) | ~1 h 40 min |
| `link_design` | 19 min |
| `opt_design` | 12 min |
| `place_design` | 2 h 34 min |
| `phys_opt_design` | 19 min |
| `route_design` | 4 h 42 min |
| reports | ~10 min |
| `write_bitstream` | 26 min |

v++ auto-scales the clock on failure: A3 exited 3 with "changed to 90.8 MHz".
The owner must decide in advance whether a scaled image counts (open question
2).

Post-route pass criteria are the same as the 90 MHz qualification
(`operating_preload_review_90mhz.json`), now at 100 MHz:
1. WNS ≥ 0, TNS = 0, WHS ≥ 0, THS = 0, 0 pulse-width failures, 0 route errors.
2. Cores 0/1/2 and local TMU 0/1/2 entirely in SLR0/1/2, with 0 unplaced.
3. CDC: every Critical reviewed by endpoint, 0 in the TNCA hierarchy. A3 had 56
   platform Criticals: CDC-11 32, CDC-7 13, CDC-10 7, CDC-13 4.
4. Unconstrained pins limited to the same 7 inputs and 3 outputs of board
   management.
5. DRC: 0 errors and 0 critical, warnings dispositioned.
6. Per-SLR LUT/FF/BRAM/URAM/DSP, CLB % and the SLL matrix reported against A3.
7. Image SHA recorded.

Loading is the owner's decision:
1. `program.sh --check` (VBNV, interface UUID, idle device, temperatures).
2. Packaged-HBM preflight.
3. `program.sh --go`, only with an explicit OK.
4. Clock-counter readback, expecting [100, 500, 450, 0].
5. `tools/r40/board_checkpoint.py`.

**Total plan time from approval:**

| Phase | Time |
|---|---|
| census | ~1 h |
| RTL changes and regressions | ~1-2 days on GB10 |
| OOC/structural audit | ~4-6 h |
| CI + archive | ~2 h |
| link | ~11 h |
| post-route audit | ~1.5 h |

## 6. Open questions for the owner

1. **Which milestone owns the new image?** v3 says R40 makes no further
   implementation or programming, and every R40 gate is bound to a88ebdf5. Is
   this implementation the deferred "next implementation together with
   frequency optimization" (outside R40 delivery), or does it need an identity
   amendment?
2. **Fallback.** If the final route misses 100 MHz and v++ scales the clock,
   is an image at a measured frequency above 90.8 MHz an acceptable
   deliverable, or is the attempt a failure?
3. **Census timing.** May the read-only census (about 1 h, about 18 GB, 2
   threads) run on the FPGA host after the current board campaign, and when?
   Other users' Vivado jobs currently hold about 40 GB.
4. **M4.** Do you accept the FPU conversion fix in the final image? It is an
   ISA conformance fix that no frozen suite uses. Or should it be an erratum,
   to keep the change set strictly timing-only?
5. **C1.** Pre-authorise including the MSHR snapshot if the census shows slice
   RAM families failing.
6. **Tool flow.** Keep A3's flow exactly (recommended), or also enable a
   post-route `phys_opt_design`? The latter is a directive-level addition the
   P0/P1 note warns against relying on.

## Files

- `plan.json`: machine-readable plan.
- `attempt03_100mhz_reported_path_groups.json`: groups of the reported 100 MHz
  paths, with coverage 0.06 % of endpoints.
- `attempt03_impl1_reports.tgz`, `attempt03_impl1_runme.log.gz`: read-only
  copies of the A3 impl_1 reports and log.
- `tools/r40/census_failing_endpoints.tcl`: full failing-endpoint census plus
  residual what-if (read-only DCP).
- `tools/r40/group_failing_paths.py`: groups census TSV or printed reports into
  cones.
- `tools/r40/audit_cross_slr_cones.tcl`: structural before/after proof on the
  synthesis DCP, with a placement-hook dry-run.
