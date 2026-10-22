# F10: registered LSU request boundary and decoupled ownership-range enables

Status: **implemented on `r41s2-io`; functional regression passes; unit OOC screen
passes the 9.0 ns projected target for both census families.** It is not a routed
result. The kernel synthesis, the routed census and board runs are still to be done.

**Responsible sources** (the base hashes match the preflight's F10 entry):

| File | Base | New |
|---|---|---|
| `rtl/core/tnca_lsu.v` | `1b27a4a9…` | `661ab3b7…` |
| `rtl/core/tnca_lsu_pair.v` | `1d0dc847…` | `a5a43c26…` |
| `rtl/noc/tnca_core_ni.v` | `ba3c39d0…` | unchanged |
| `rtl/noc/tnca_router.v` | `b190021b…` | unchanged |
| `rtl/nuca/tnca_cache_handoff.v` (the second family's endpoint) | `1163ab95…` | `25636165…` |

**Bench:** `tb/core/tb_lsu_reqreg.v`, run by `make lsu-reqreg`. **OOC harness:**
`flow/fpga/tnca_lsu_mesh_ooc.v`. The NI and router needed no change: the boundary is
placed where the long path starts.

## The census families

From the A3 routed census (`evidence/r40/v3/final_image_candidate/census/`):

| Family | Endpoints | TNS | Worst | Worst path |
|---|---:|---:|---:|---|
| LSU `pending` → mesh router local request FIFO (`u_r/g_port[0].g_vn[0].u_in/mem_reg` LUTRAM write data) | 183 | −75.35 ns | −0.881 ns | tile 0 unit 1, 16 levels, 10.473 ns: 1.613 ns logic, 8.860 ns route. `pending` → lowest-lane select → `plan_word` mux → 64-bit word compares → `plan_lanes` → write-data AND-OR → pair/mem/host mux → NI flit → FIFO |
| LSU `FSM_sequential_state` → `g_cache_handoff.ownership/home_base`, `home_size` **CE** | 128 | −66.70 ns | −0.818 ns | SLR0 → SLR1, 7 levels, 10.095 ns: 0.964 ns logic, 9.131 ns route. LSU state → `core_idle` → `&tile_idle` → handoff `ready` → 128 enables |

In the first family, the planner's own `pending` → planner segment is about 8.9 ns of
the routed 10.47 ns. Registering at the NI or router would therefore leave almost all
of it, so the boundary goes inside the LSU, right after the planner.

## Design

### 1. LSU request boundary (`REQ_REG`)

- **Parameter.** `tnca_lsu` takes a new `REQ_REG` parameter, default 0, which keeps the
  R40 port. `tnca_lsu_pair` passes `REQ_REG = 1` (its new default) to both units. It
  therefore applies to every `LSU_INFLIGHT = 2` configuration: the multicore image, the
  `core-vrf-sync-v3/v4/v5` profiles and the multicore shells. `tnca_core.v` is untouched.
- **Queue.** Each unit gets a two-entry FIFO (head and tail registers) of planned beats.
  An entry holds the word, the beat tag and two lane masks: lanes served by their low
  half, and lanes served by their high half.
- **Credit.** A beat is planned and issued into the queue when the tail is free, which is
  a local credit. It no longer waits for the port's ready. At that push, the beat-table
  slot, `pending`/`partial` and `bt_*` update exactly as they did at the port transfer
  before.
- **Port.**
  - `mem_req_valid`, `addr` and `tag` are queue registers.
  - `be` and `wdata` are formed at the head, by an AND-OR of its registered lane masks
    with the S_SETUP lane geometry (`ln_lo/hi_be`, `ln_lo/hi_wd`). That geometry is
    captured once per access.
  - The head pops on `mem_req_ready`.
  - `we`, `hint`, `amo`, `aop` and `cmp` remain per-access registers.
- **S_RUN exit.** The unit leaves S_RUN on the edge its queue empties.

### 2. Ownership range (`tnca_cache_handoff`)

`home_base` and `home_size` load `base`/`size` on every idle cycle (`!busy`) and hold
while busy. Their 128 enables become the local `busy` register instead of the global
quiescence tree carried by `ready`.

## Correctness argument

- **Same request stream.** The planner and its bookkeeping are unchanged and run at the
  same point (the issue, now into the queue). The beat sequence of an access is
  therefore identical. The queue is FIFO, and `be`/`wdata` are a function of the lane
  masks and of geometry fixed for the whole access, so the port sees the same beats in
  the same order.
- **Backpressure.**
  - A stalled head holds valid and payload. The head and tail registers change only on
    pop or push, and the geometry is fixed. This is the `tnca_fifo` handshake rule.
  - With the tail full, issue stops (no credit) and `pending` holds.
  - The port's ready reaches only the pop, the head/tail move and the S_RUN exit. Before,
    it reached `pending` and the beat table through `issue_fire`.
- **Ordering and single issuer (`tnca_lsu_pair`).**
  - `o_draining` (S_DRAIN/S_DONE) is entered only on the edge the queue empties. It
    still implies that every request of the access has reached the port.
  - The second unit is accepted only on `o_draining` of the first. So only one unit's
    queue is ever non-empty, `mu = u_mvalid[1]` stays a correct port select, and the
    earlier access's requests precede the later one's.
  - An atomic still waits for both units to be idle.
- **Response ownership.** A beat's slot is reserved at the push, before the request can
  leave. A response can only follow its request, so it always finds its `bt_valid`
  entry, and the response path stays unconditionally ready. Tags remain unique while
  outstanding, because `free_tag` comes from `bt_valid` as before.
- **Fault drain.** The LSU has no kill. An accepted access always completes, queued beats
  included, and the unit reports idle (`req_ready`, `core_idle`) only from S_IDLE, when
  the queue is empty.
  - Fences and arrives still mean every access is complete.
  - Host grants and cache maintenance still see a non-idle core until the queue has
    drained and every response has returned.
- **Ownership range.**
  - The value captured at the accepting edge is the same: `!busy` holds on that edge.
  - The value is held from acceptance until done.
  - While idle, the range changes, but nothing reads it then. `home_valid = busy &
    ~issued` gates the command links. `tnca_fifo` writes only on push. The slices sample
    `maint_base`/`maint_size` only on `maint_valid && maint_ready`.

## Verification on GB10

Tree: `r41s2` + F06 + F10, as `~/tritonchip-r41s2-io-f10`, with `python/tnca/data`
restaged. The base comparisons use the unchanged `r41s2` tree. Logs are in
`f10-regression-logs.tgz`.

**New directed miter, `make lsu-reqreg` (`tb_lsu_reqreg.v`).** The target is also added to the CI job that runs the dispatcher benches.
- **Setup.** Two `tnca_lsu_pair` instances, `REQ_REG = 0` and `REQ_REG = 1`, run the
  same random program. It covers scalar, vector and indexed loads and stores of every
  size and alignment, straddling lanes, masks and 32/64-bit atomics. Each instance has
  its own memory, which performs requests on acceptance and adds random back-pressure,
  long stalls and responses delayed by 1–24 (or 1–200) cycles, in random order.
- **Required equal:**
  - the request stream (tags excepted);
  - the completions, filed by access, since the pair may complete the younger first;
  - the final memory.
- **Protocol checks on the registered instance:**
  - a stalled request holds valid and payload;
  - never both units on the port;
  - never draining, or idle, with queued beats;
  - every response tag owned by an outstanding beat of the unit it names;
  - no tag reused while outstanding.
- **Runs:**
  - seeds 1–4: 30 % stall; 60 % stall; no stall with in-order responses; 200-cycle
    latency, each with 1,500 accesses and 3,800–4,100 requests;
  - a directed program checking one beat per cycle and exactly +1 cycle of request
    latency per access.

  All pass.
- **Mutations**, each caught:
  - no credit (queue overrun);
  - S_RUN exit without an empty queue;
  - wrong half-select in the head;
  - tail not cleared on move-up;
  - pop without ready.

**Regression targets** (F10 tree):

| Target | Result |
|---|---|
| `make lsu-reqreg` | PASS (5 runs) |
| `make core-r39sw5` (v5 profile, `LSU_INFLIGHT = 2`, every core program incl. faults `t05/t17/t23/t32`, atomics `t22`, straddles `t37`, descriptors, xpose, pairs; in-order and random-stall/reordering memory) | PASS (86) |
| `make core-r39sw4` / `make core-r39sw` | PASS |
| `make core`, `make core-ooo` (`LSU_INFLIGHT = 1`, legacy port) | PASS |
| `make top`, `make mesh`, `make nuca` | PASS |
| `make r40-multicore-shell` (shell defaults `LSU_INFLIGHT = 2`) | PASS |
| `make r40-line-axi` (multicore AXI, N = 1–8, deadline and physical-fault drain, write delay 300) | PASS |
| `multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top` | PASS |
| `make r40-cache-maintenance` (`tb_cache_handoff` homes 1/3/8, slice maintenance, global handoff on real cores) | PASS |
| `make r40-multicore-dma`, `make r40-grid-dispatch`, `make r40-coherent-host` | RESULT_DMA |
| Native Torch on the board-top multicore RTL (`native_multicore_features.py --backend fpga-multicore-sim --cores 3 --pipeline 2 --board-top`, with and without `--pair`) | RESULT_NATIVE |

## Cycle impact

Each LSU access in an `LSU_INFLIGHT = 2` configuration pays one more cycle between
acceptance and its first request at the port. The unit reports draining one cycle
later, so the next access in the pair also starts one cycle later. The beat rate is
unchanged at one per cycle.

| Workload | Base | F10 | Change |
|---|---:|---:|---:|
| Directed miter: scalar, 8-beat, 9-beat and 16-beat accesses | first request at +2 cycles | +3 cycles | +1 cycle per access, same span |
| Miter, no stall, 1,500 random accesses (memory-bound stream; worst case) | 21,668 | 22,450 | +3.6 % |
| `core-r39sw5`, all 86 program × memory-model runs, total | 246,160 | 247,740 | **+0.64 %** |
| `core-r39sw5`, per program | | | −2.3 % to +8.7 %. The largest are short or memory-bound programs: `t18_squash` +8.7 %, `t04_roles` ooo +8.6 %, `t23_amofault` ooo +7.1 %, `t16_vecadd` ooo +5.0 %, `t22_atomic` ooo +4.8 % |
| `core-r39sw4`, all runs, total | 236,129 | 237,760 | +0.69 % (worst `t18_squash` +8.7 %) |
| `core-r39sw`, all runs, total | 225,666 | 227,453 | +0.79 % (worst `t18_squash` +8.7 %) |
| `tb_multicore_axi` (`r40-line-axi`), 455-program grid on real cores and AXI outlets, grid span (F06 + F10) | N = 1: 39,033; N = 3: 13,659; N = 8: 8,600 | 39,492; 13,809; 8,675 | +1.2 %, +1.1 %, +0.9 % |
| `tb_multicore_shell` (`r40-multicore-shell`), same grid, N = 3 | 12,990 | 13,143 | +1.2 % |
| NATIVE_ROW | | | |

## Unit OOC timing at 10 ns

- **Run:** `vivado_ooc_timing.tcl`, `OOC_SV=1`, part `xcu280-fsvh2892-2L-e`, at the
  image parameters: `VLEN=16 ELEN=32 RCTX_W=2 NUM_BEATS=8 CTAG_W=4`, tag 5, `NX=3 NY=1`,
  `XW=YW=2`, NoC depth 4, `N=3`.
- **Where:** FPGA host, 2026-10-21, through `vivado_slot.sh`.
- **Harness:** `tnca_lsu_mesh_ooc`. It chains, as `tnca_core`, `tnca_tile` and `tnca_top`
  connect them:
  - the real `tnca_lsu_pair`;
  - `tnca_mem_mux`;
  - tile 0's `tnca_idle_host_mux`;
  - `tnca_core_ni`;
  - a real `tnca_router`, whose local request FIFO is the census endpoint;
  - the real `tnca_cache_handoff`, quiesced by the LSU's idle.

  Every other input and output is registered.
- **Raw files:** reports, family queries and exact sources are in
  `f10-ooc-evidence.tgz`.

| Unrouted estimate, 10 ns | Base | F10 |
|---|---:|---:|
| WNS | +5.365 ns | +6.677 ns |
| Family 1: `pending` → mesh FIFO | `pending_reg[3]` → `u_in/mem_reg` RAM input, 18 levels, **4.493 ns** | no such path. LSU → FIFO: `q_hv_reg` → FIFO header bits, 6 levels, **1.748 ns** |
| Family 1, the planner segment now ending in the LSU | `pending` → `bt_lanes` RAM, 12 levels, 3.092 ns | `pending_reg[7]` → `pending_reg[15]`, 15 levels, **3.305 ns** (the worst path of the unit) |
| Family 2: LSU FSM → `home_base`/`home_size` | `FSM_sequential_state_reg[0]` → `home_base_reg[0]`, 4 levels, **1.243 ns** | no such path. Into those registers: `busy_reg` → `home_base_reg`, 1 level, **0.524 ns** |
| LSU FSM → other ownership state (not a census family) | → `completed_reg`, 5 levels, 1.534 ns | → `completed_reg`, 4 levels, 1.320 ns |
| CLB LUTs / registers / LUTRAM (harness: 2 LSU units, 1 router) | 28,196 / 18,430 / 1,352 | 28,877 / 18,784 / 1,352 |

**Projection** (after-synthesis × census routed / baseline synthesis, per family):
- **Family 1.** Ratio 10.473 / 4.493 = 2.331.
  - The worst new path of the family, `pending` → `pending` at 3.305 ns, projects to
    **7.70 ns**.
  - The new LSU → FIFO path projects to 4.07 ns.
  - Both are within 9.0 ns.
  - The ratio comes from a path that is 85 % route between the LSU and the mesh. For an
    LSU-internal path it is likely pessimistic.
- **Family 2.** Ratio 10.095 / 1.243 = 8.12, set by the SLR crossing and the fan-out to
  128 enables.
  - The endpoints' new driver, `busy` → enable, projects to **4.26 ns**. The census
    endpoints no longer have a cross-SLR path at all.
- **Resources.** +681 LUTs and +354 registers per core: the head formation and two
  queue entries per unit. That is about +0.16 % of an SLR's LUTs in each of the three
  SLRs.

## Open risks

- **Residual quiescence cone.** The cross-SLR path from the LSU FSM through `core_idle`
  and `&tile_idle` still exists, into the handoff's few control registers (`busy`,
  `issued`, `completed`). It also reaches, via `all_idle`, the MMIO `control`
  registers: two residual endpoints at −0.138 and −0.054 ns, 10 levels, not in F10's
  listed families.
  - The handoff control endpoints met timing in A3.
  - Applying family 2's ratio to their 1.320 ns would give 10.7 ns, but that ratio is
    dominated by the 128-enable fan-out the fix removes.
  - Cutting this cone needs a registered per-tile idle. Its staleness would have to be
    covered by the dispatcher's `core_busy` (live, local) and a registered launch flag.
    That is a separate candidate touching `tnca_tile`/`tnca_top`.
- **Packing.** The `REQ_REG` logic adds LUTs in SLR0, which A3 packed to 95.24 %.
- **Scope of `REQ_REG`.** `REQ_REG = 1` is the pair's default, so every
  `LSU_INFLIGHT = 2` simulation profile changes cycle counts too, not just the image.
  The compiler's scheduler model (`spec/r39_sched_model.json`, LSU base latency) was not
  updated. It is a performance heuristic, and any change to it would move compiler
  output.
- **Projection basis.** The projections use one routed census, and both families were
  net-dominated. Only a routed run shows whether the router FIFO's placement relative to
  the core holds.
