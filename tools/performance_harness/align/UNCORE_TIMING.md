<!-- Status: DRAFT, written 2026-10-27 by reading the Verilog at 1f82d54 (an agent's pass, reviewed
     for structure only). The working reference for the cycle-level uncore; the alignment gate is the
     authority and section 8 lists what reading could not settle. -->

# TNCA uncore cycle-timing specification (draft, from the Verilog)

Worktree `TritonChip-model-align`, branch `model-align`, commit `1f82d54`.
Everything between a core's `mem_req_*`/`mem_rsp_*` port and the DRAM port of
the device: idle host mux, core NI, mesh, slice NI, NUCA slice, home link,
memory arbiter/ports, DRAM model (simulation) or AXI outlet (board), plus the
dispatch and host paths. Line numbers are of the files named in each section.

## 0. Conventions, and the two configurations that matter

Cycle `t` is the interval ending at clock edge `E_t`. A register assigned with
`<=` at `E_t` holds the new value during `t+1` (same as CORE_TIMING.md sec 0).
"Accepted at `t`" means `valid && ready` during `t`; the transfer is committed
at `E_t`. Every FIFO in this path is `tnca_fifo` (`rtl/common/tnca_fifo.v`):
`in_ready = (cnt < DEPTH)` (32), `out_valid = (cnt != 0)` (33), push and pop
are registered (46-56). A datum pushed at `E_t` is at the head during `t+1`;
a full FIFO being popped during `t` still refuses a push during `t` (no
pop-through, no bypass). Minimum transit of one FIFO = 1 cycle.

There are TWO uncore configurations and they differ in cycle behaviour:

| parameter | board (U280 image, `evidence/r40/v2/convergence/attempt03/image.tcl`) | simulation bench (`tb/device/tnca_multicore_device.v` via `python/tnca/backends/multicore_file.py`) |
|---|---|---|
| NX x NY | 3 x 1 (`rtl/board/tnca_multicore_hbm_xo.v:337`) | `NX = N>4 ? 4 : N`, `NY = N/NX` (device.v:7): 3 x 1 for N=3 |
| SETS / WAYS | 32 / 8 (image) | **4 / 2** (`device.v:37`: `.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)`), not overridable (`multicore_file.py:35` allows only core flags) |
| FULL_LINE_TAG | 1 (`tnca_tile.v:336`: `!(NX==2 && NY==2)`) | 1 |
| NOC_LINK_STAGES | 2 | 0 (shell default, `tnca_multicore_shell.v:13`) |
| MEM_HOME_STAGES | 2 | 0 |
| DISPATCH_STAGES | 2 | 0 |
| CODE_LINK_STAGES | 2 | 0 |
| MEM_PORTS | 3 (`tnca_mem_ports`) | PORTS=1 in our recordings (`tnca_mem_arbiter`), `tnca_dram_model` LATENCY 11, RANDOM_STALL 1, REORDER 1 |
| MEMORY_CONTROLS | 1: `tnca_memory_mode` + dynamic AXI admission (`tnca_multicore_axi.v:258-287`) | n/a |
| CACHE_MAINTENANCE | 1 | 0 (shell default) |
| NUCA_REGISTER_VICTIM_SELECT / _RANK / CACHE_DATA_PIPELINE | 1 / 1 / 1 | 0 / 0 / 0 |
| REGISTER_VICTIM_OWNER | 1 (hard-wired `tnca_tile.v:335`) | 1 |
| DCO_REGISTERED_RANGES / DCO_LOCAL_REPLICAS / DCO_REGISTER_WRITES / DCO_TLW_PIPE_STAGES | 1 / 1 / 1 / 2 | 0 / 0 / 0 / 0 |
| MSHR_N / TARGET_N / RQ_N / NOC_DEPTH | 8 / 16 / 8 / 4 (`tnca_top.v:71-78`) | same |
| slice RSPQ_N | 4 (`tnca_nuca_slice.v:496`) | 4 |
| HOST_ACCESS | tile 0 only (`tnca_top.v:452`) | same |
| MEM_SRC_W | 2 (`N>4?$clog2(N):2`) | 2 |
| XW / YW | 2 / 2 (`tnca_top.v:38-39`) | same |

The model-side `r40-board-m2.cfg` (`sets=32`, `ways=8`, `noc_stages=3`,
`outlets=3`, `outstanding=8`) describes the BOARD. Any alignment against the
simulation bench must use sets=4, ways=2, hop=1, no home link, no outlet
(sec 9).

Reset-state facts that fix the policy on both configurations unless a program
or the host writes the TMU window (`TMU_BASE = 0x7FFF_0000`, 4 KiB,
`tnca_top.v:84`): `policy <= 8'd0` (`tnca_tmu.v:303`) means `pol_repl = 0`
(round-robin victim), no dead-block prediction, no bypass, no tile-last-word
notices (`tnca_nuca_slice.v:232-238`, 854-858, 888-889). The compiler CAN emit
TMU writes (`compiler/python/tnca_cc/dco.py:485`); see sec 8 item 1.

## 1. End to end: a core request to DRAM and back

### 1.1 Stage table (what each hop costs, in order)

| # | stage | file:lines | cost | refuses when |
|---|---|---|---|---|
| 1 | `tnca_mem_mux` / LSU (core side) | CORE_TIMING sec 3 | core's business | - |
| 2 | idle host mux (tile 0 only) | `tnca_idle_host_mux.v:21-33` | 0 (combinational pass-through: `n_valid = c_valid`, `c_ready = n_ready && !use_host`) | `use_host` (never during a launch, sec 7) |
| 3 | core NI inject | `tnca_core_ni.v:106-108` | 0 (`net_in_valid_req = req_valid`, `req_ready = net_in_ready_req`) | - |
| 4 | source router local VN_REQ input FIFO | `tnca_router.v:77-84`, DEPTH 4 | 1 (push `E_c`, head `c+1`) | `cnt == 4` -> `mem_req_ready = 0` at the core |
| 5 | per hop router->router | router pop + `tnca_elastic_link` STAGES | `1 + NOC_LINK_STAGES` (3 board, 1 sim) | downstream FIFO full |
| 6 | home router local VN_REQ output -> slice NI -> slice `req_valid` | `tnca_slice_ni.v:65-66`, `tnca_nuca_slice.v:393` | 0 (combinational) | `!rq_has_free` (8 entries) or maintenance |
| 7 | request buffer -> engine E_SEL | slice 1069-1082 | >= 1: accepted `E_a`, selectable during `a+1` | engine not in E_SEL, `ms_pick`, `rsq_full`, blocked/shadowed |
| 8 | engine: hit | E_SEL, E_LOOK, E_HIT | 3 engine cycles; `rsp_valid` 1 cycle after E_HIT | - |
| 8' | engine: miss to memory request | sec 3.4 | sim: SEL,LOOK,MISS, SEL,MREQ (5); board: +ALLOC (6); +3 (SEL,WBRD,WBSND) for a dirty victim | `mem_req_ready = 0` holds the engine in E_MREQ/E_WBSND |
| 9 | home link (board only) | `tnca_line_link.v` STAGES 2 | +2 request, +2 response | stage-0 FIFO full (2 entries) |
| 10 | memory arbiter / ports / mode | `tnca_mem_arbiter.v`, `tnca_mem_ports.v`, `tnca_memory_mode.v` | 0 (combinational both ways) | round-robin loss, `held` |
| 11 | outlet | sim: `tnca_dram_model.v`; board: `tnca_line_link` (LINK_STAGES 2) + `tnca_line_axi.v` | sim: response `LATENCY+1 = 12` after accept; board: +2/+2 link, 2/3-cycle issue, HBM, 2 beats, +1 | sim: `lfsr[5] == 0`; board: admission/hazard/issue busy |
| 12 | slice fill and wake | sec 3.5 | response at `r` -> SEL,FILL,SEL,WAKE -> `rsp_valid` at `r+5` (first target), +2 per further target | engine busy elsewhere, `rsq_full` |
| 13 | slice response queue -> slice NI -> home router local VN_RSP input FIFO | slice 504-510, `tnca_slice_ni.v:91-93` | 1 (push `E_t`, head `t+1`) | router FIFO `cnt == 4` (response stays in `rsq`, 4 deep) |
| 14 | per hop (VN_RSP) | as 5 | `1 + NOC_LINK_STAGES` | downstream FIFO full |
| 15 | core router local VN_RSP output -> core NI register | `tnca_core_ni.v:120-129` | 1 (`rsp_valid <= net_out_valid_rsp`; `net_out_ready_rsp = 1` always) | never |

### 1.2 Reference timelines (no contention, engine idle, `rq`/`rsq` empty)

Let `c` = cycle the core's request is accepted (`mem_req_valid && mem_req_ready`),
`H` = number of mesh hops between the core's tile and the home, `LS` = NOC_LINK_STAGES.

| event | cycle |
|---|---|
| flit at head of source router local FIFO | `c+1` |
| flit at head of home router (local output valid) = slice `req_valid` | `a = c+1+H*(1+LS)` |
| slice accepts into `rq` | `E_a` |
| E_SEL selects it (`s`) | `s = a+1` |
| E_LOOK / E_HIT | `s+1` / `s+2` |
| slice `rsp_valid` (hit) | `s+3 = a+4` |
| head of home router local VN_RSP FIFO | `a+5` |
| head at core's router local output | `a+5+H*(1+LS)` |
| core `mem_rsp_valid` (hit) | `a+6+H*(1+LS)` = `c+7+2H(1+LS)` |

Hit: local `c+7`; sim 1 hop `c+9`, 2 hops `c+11`; board 1 hop `c+13`, 2 hops `c+19`.

Miss, simulation bench (PORTS=1, DRAM LATENCY 11), no dirty victim:

| event | cycle |
|---|---|
| E_MISS (allocate MSHR, `MS_REQ`) | `s+2` |
| E_SEL picks the MSHR | `s+3` |
| E_MREQ, `mem_req_valid`; fires if arbiter grants and `lfsr[5]` | `m = s+4` (first opportunity) |
| DRAM `rsp_valid` (= slice `mem_rsp_valid`, arbiter is combinational) | `r = m+12` |
| E_SEL picks `MS_FILL` | `r+1` |
| E_FILL | `r+2` |
| E_SEL picks `MS_WAKE` | `r+3` |
| E_WAKE (push response) | `r+4` |
| slice `rsp_valid` | `r+5` |
| core `mem_rsp_valid` | `r+7+H(1+LS)` |

Uncontended local miss on the sim bench: `s = c+2`, `m = c+6`, `r = c+18`,
core response at `c+25`. Board: E_ALLOC adds one engine cycle (`m = s+5`), the
home link adds 2 each way, LINK_STAGES adds 2 each way, the AXI adapter and HBM
add their own (sec 4.3).

### 1.3 Back-pressure that reaches `mem_req_ready`

`mem_req_ready` at the core = `in_ready` of the source router's local VN_REQ
FIFO (`tnca_mesh.v:78`, `tnca_router.v:79`). That FIFO drains only when its
head flit is granted its output port (`pop`, router 153-161). The chain, from
the slice upward: slice `req_ready = rq_has_free && !maint_busy && !maint_requested`
(slice 393) -> home router local VN_REQ output not popped -> home router's input
FIFO for that direction fills (4) -> link stage FIFOs fill (2 per stage) ->
upstream router's output not popped -> ... -> source FIFO fills (4) ->
`mem_req_ready = 0`. Nothing else drives `mem_req_ready`. A request is never
dropped or retried on the network.

## 2. The NoC

### 2.1 Topology and ports

`tnca_mesh.v`: node index `y*NX + x` (53); router ports 0 local, 1 north (+y),
2 east (+x), 3 south (-y), 4 west (-x) (40). Links are `tnca_elastic_link
#(.W(FW), .STAGES(LINK_STAGES))` per direction per VN (88-91, 103-106,
113-116, 128-131). Edge ports: `r_in_valid = 0`, `r_out_ready = 1` (93-99,
118-124). For 3x1 only east/west links exist: router 0 E <-> router 1 W,
router 1 E <-> router 2 W.

Two virtual networks (`tnca_noc_pkg.vh:16-18`): VN 0 = VN_REQ, VN 1 = VN_RSP.
Each (port, VN) has its own input FIFO (router 73-87) and its own output
arbiter (128-144). A tile injects requests only on VN_REQ (core NI) and
responses only on VN_RSP (slice NI); a tile ejects VN_REQ into its slice and
VN_RSP into its core (`tnca_tile.v:381-395`).

### 2.2 Home mapping (`tnca_core_ni.v:83-91`)

```
line   = req_addr >> 6
hash   = line[7:0] ^ line[15:8] ^ line[23:16] ^ line[31:24] ^ line[39:32] ^ line[47:40] ^ line[55:48] ^ line[63:56]
home   = (NX==2 && NY==2) ? {6'd0, line[1:0] ^ line[3:2]} : hash % (NX*NY)
home_x = home % NX ;  home_y = home / NX
```
For 3x1: `home = hash % 3`. The function takes the address only (no hint,
no descriptor). Response destination: `dst_x = rsp_src % NX`, `dst_y = rsp_src / NX`
(`tnca_slice_ni.v:88-89`), `rsp_src` = requester's tile id carried in the
request (`TNCA_REQ_SRC = MY_ID[3:0]`, core NI 99).

### 2.3 Packet format: exactly one flit per packet

`flit = {dst_y[YW-1:0], dst_x[XW-1:0], payload[215:0]}`, FW = 2+2+216 = 220
bits (`tnca_noc_pkg.vh:7,15`; `tnca_top.v:175`).

| field | bits | request (VN_REQ) | response (VN_RSP) |
|---|---|---|---|
| ADDR | 63:0 | byte address (word granular) | RDATA 63:0 |
| WE | 64 | | TAG 68:64 |
| BE | 72:65 | | 215:69 zero |
| WDATA | 136:73 | | |
| SRC | 140:137 | requesting tile | |
| TAG | 145:141 | requester tag (bit 4 = tensor engine, low 4 = client tag, `tnca_noc_pkg.vh:31-34`) | |
| AMO / AOP / CMP | 146 / 150:147 / 214:151 | | |
| 215 | 0 | | |

Every request and every response is one word (8 bytes) in one flit; no line
ever crosses the mesh. Link width accounting does not exist in the RTL
(`tnca_router.v:22-25`); the model's 32/16 bytes per flit are its own
bookkeeping.

### 2.4 Router (`tnca_router.v`)

| rule | expression / lines |
|---|---|
| input FIFO per (port, VN) | DEPTH = NOC_DEPTH = 4, AW 2 (77-84) |
| route (X then Y then local) | `dx > X_HERE -> EAST; dx < X_HERE -> WEST; dy > Y_HERE -> NORTH; dy < Y_HERE -> SOUTH; else LOCAL` (104-112), from the head flit's header |
| output arbitration per (out port, VN) | round robin over input ports; candidates `(rr[oc] + k) % 5` for k = 0..4, k = 0 has priority (loop 134-141 runs k from 4 down to 0, last assignment wins) |
| `out_valid[oc]` | `= sel_ok[oc]` (165): independent of `out_ready` |
| pop | `pop[sel_in*NV+vn] = sel_ok && out_ready` (153-161): transfer in the cycle both are high |
| pointer update | at the transfer edge `rr[oc] <= (sel_in == 4) ? 0 : sel_in + 1` (176-178); reset 0 (174) |
| per-flit latency through a router | 1 cycle (input FIFO); no output register |
| head-of-line blocking | yes: one FIFO per (input port, VN), only the head is considered |

An input port's head flit can go to exactly one output, so no two outputs
select the same input (147-150).

### 2.5 Links (`tnca_elastic_link.v`)

STAGES chained `tnca_fifo #(.DEPTH(2))` (19-25). Each stage adds exactly 1
cycle when empty and holds 2 flits; `in_ready` of a stage depends only on that
stage's count. STAGES = 0 is a wire (12-17). Per hop, from "head at router A"
to "head at router B": `1 + STAGES` cycles (board 3, sim 1); buffering per hop
per VN: 4 (router FIFO) + 2*STAGES.

### 2.6 Contention

Two flits of the same VN wanting the same output port in the same cycle: the
one at `rr` (or nearest after it) wins; the loser waits a cycle and the pointer
has moved past the winner, so two steady streams alternate. Request and
response never compete (separate FIFOs and arbiters per VN). A stalled VN_REQ
never blocks VN_RSP. Local injection at a router competes with pass-through
traffic on the same output under the same round robin (port 0 is one of the
five candidates).

## 3. The NUCA slice (`rtl/nuca/tnca_nuca_slice.v`)

### 3.1 Geometry

| item | value | lines |
|---|---|---|
| line | 64 B, 8 words | 183-185 |
| set index | `addr[6 +: SET_W]` (FULL_LINE_TAG = 1 -> HOME_SKIP = 0) | 190, 212 |
| tag | `addr[63 -: TAGB]`, TAGB = 64 - 6 - SET_W | 191, 213 |
| board | 32 sets x 8 ways = 16 KiB per slice; set = addr[10:6] | image |
| sim bench | 4 sets x 2 ways = 512 B per slice; set = addr[7:6] | device.v:37 |
| request buffer `rq_*` | 8 entries, not a FIFO | 361-378 |
| MSHRs | 8 (`ms_*`), targets 16 (`tg_*`), one target list per MSHR | 453-476 |
| response queue `rsq_*` | 4 entries | 496-510 |
| tag banks | one LUTRAM bank per way, synchronous read (SYNC_TAG_BANK = 1) | 596-608 |

### 3.2 Request buffer and arbiter (every cycle, combinational)

| rule | lines |
|---|---|
| accept: `req_ready = rq_has_free && !maint_busy && !maint_requested`; entry = lowest free index; `older[new] = 0`, `older[i][new] = rq_v[i]` | 380-393, 1022-1038 |
| eligible: `rq_v[i] && !rq_blocked[i] && !shadowed`, shadowed = an OLDER live entry (blocked or not) to the same line `addr[63:6]` | 405-416 |
| selected: the oldest eligible (`older` matrix) | 418-432 |
| freed: at E_HIT, E_MMIO completion, and in E_MISS/E_ALLOC on merge, bypass or allocation (`rq_v[cur_rq] <= 0`) | 1142, 1216, 1288, 1315, 1354 |
| blocked: set in E_MISS/E_ALLOC when no MSHR/target/way, or a store meets a bypassed miss; cleared for ALL entries when any MSHR serves its last target (`rq_blocked <= 0` in E_WAKE) | 1266, 1291, 1361, 1466 |

Consequence: same-line requests are serialised in arrival order; a younger
same-line request waits until the older one has LEFT the buffer (merged,
allocated, or hit), not merely started.

### 3.3 Engine state machine (`est`, one engine per slice, 520-535)

E_SEL decision order (1054-1082): maintenance (board only) > `ms_pick` > new
request. `ms_pick` (895-917): MS_WAKE > MS_FILL > MS_WB > MS_REQ, and within a
state the LOWEST MSHR index (loops run high to low, last match wins). A new
request is selected only when `sel_valid && !rsq_full` (1069). The tag banks
are read in E_SEL for the selected entry's set (590-593, 604) and consumed in
E_LOOK.

E_LOOK (1121-1131): `cur_mmio -> E_MMIO`; `hit && !ms_match -> E_HIT`; else
`E_MISS`. `hit` = some way valid with equal tag (626-632); `ms_match` = any
non-free MSHR on this line (637-644). Note a line whose MSHR is in MS_WAKE
(filled, `v_bits` already 1) still goes to E_MISS and merges.

| path | states (each 1 cycle unless stalled) | engine cycles | response pushed at |
|---|---|---|---|
| hit (load, store, atomic) | SEL, LOOK, HIT | 3 | `E` of HIT (1175, 1197, 1202) |
| TMU register load | SEL, LOOK, MMIO | 3 | `E` of MMIO |
| TMU register store | SEL, LOOK, MMIO (waits `tmu_wr_ready`) | 3+ | `E` of the MMIO cycle in which `tmu_wr_ready` |
| miss, sim | SEL, LOOK, MISS | 3 | later (WAKE) |
| miss, board (REGISTER_VICTIM_SELECT) | SEL, LOOK, MISS, ALLOC; with `pol_repl != 0` and no invalid way: MISS, RANK, ALLOC | 4 (5) | later |
| merge into an in-flight miss (MSHR match, target free) | same as miss | 3 (4) | later |
| memory request per MSHR | SEL, MREQ (MREQ repeats until `mem_req_ready`) | 2+ | - |
| dirty victim per MSHR | SEL, WBRD, WBSND (WBSND repeats until `mem_req_ready`) | 3+ | - |
| fill per MSHR | SEL, FILL | 2 | - |
| wake per target | SEL, WAKE (WAKE repeats while `rsq_full`) | 2+ per target | `E` of each WAKE |
| epoch mismatch (TMU table commit in that cycle) | MISS/ALLOC -> LOOK, repeat | +2 | - |

A store hit writes the data array and sets `d_bits` at the HIT edge (1183-1196);
a load hit reads it (CACHE_DATA_PIPELINE = 1 on the board moves the array read
to E_LOOK into `hit_word_q`, 1124-1127; same cycle count). Atomics compute and
write in the same HIT/WAKE cycle (1150-1182, 1415-1434).

### 3.4 Miss allocation (E_MISS sim / E_ALLOC board, 1262-1363)

Priority of the branches, evaluated once, at the allocation cycle's edge:

| condition | effect |
|---|---|
| `ms_match && ms_byp[..] && (cur_we || cur_amo)` | block the entry (1266) |
| `ms_match` and `tg_has_free` | append target (lowest free index) to that MSHR's list; `rq_v` cleared; `perf_merge` (1269-1288) |
| `ms_match` and no target | block (1291) |
| no match, `ms_has_free && tg_has_free && allocation_bypass` | bypass miss (only with DCO policy bits; never at reset policy) (1293-1315) |
| no match, `ms_has_free && tg_has_free && allocation_ok` | allocate MSHR (lowest free), claim way, `v_bits[way] <= 0`, `victim_ptr[set] <= way + 1`, state `MS_WB` if the way was valid and dirty else `MS_REQ`; one target; `rq_v` cleared (1316-1354) |
| otherwise | block (1355-1362) |

Victim (policy 0 = reset): invalid way first, lowest index (790-793); else the
first way not claimed by another non-bypass MSHR of this set, scanning
`victim_ptr[set], victim_ptr+1, ...` (794-800; `way_claimed` 698-707). With all
ways claimed (sim: 2 ways, so a third concurrent miss to a set) the request is
blocked until an MSHR retires. Board with REGISTER_VICTIM_SELECT = 1 and
`pol_repl == 0`: `shortcut_use = 1` (732), E_MISS registers the shortcut
choice and goes to E_ALLOC (1226-1232); E_ALLOC allocates with the registered
`selected_*` (545-548). LRU `age` and `touch` (949-959, 1208, 1401) are
maintained but unused at policy 0.

MSHR lifetime: `MS_WB -> MS_REQ -> MS_WAIT -> MS_FILL -> MS_WAKE -> MS_FREE`.
`MS_FILL` is set by `mem_rsp_valid` at `E_r` regardless of the engine state
(1043-1046); the engine notices at its next E_SEL. `MS_FREE` at the WAKE of the
last target (1461-1466); `tg_v` cleared per target (1460). Targets are served
in allocation order (list head to tail, 1410-1471), one per SEL+WAKE pair.

### 3.5 Memory side of the slice

`mem_req_valid = est ∈ {E_MREQ, E_WBSND, E_MSEND}`, combinational (920-925);
`mem_req_we` for WBSND/MSEND; `mem_req_tag = cur_ms` (MSHR index);
`mem_req_wdata = wb_buf` (loaded in E_WBRD from the data array, 1369). While
`mem_req_ready = 0` the engine stays in E_MREQ / E_WBSND and serves NOTHING
else (1374-1388). `mem_rsp_*` is consumed unconditionally (no ready): the
line goes to `ms_line[tag]` and the state to MS_FILL (1043-1046).

### 3.6 Hit-under-miss, miss-under-miss, merging

| situation | behaviour |
|---|---|
| hit to another line while an MSHR waits | normal SEL/LOOK/HIT; the engine is free during MS_WAIT |
| miss to another line | its own MSHR (up to 8), its own SEL+MREQ; memory requests serialise at the arbiter |
| second request to a line with an MSHR | E_MISS/E_ALLOC merge: 3 (sim) / 4 (board) engine cycles now, 2 more (SEL+WAKE) after the fill, served after the earlier targets |
| request to a line whose MSHR is MS_WAKE | still merges (ms_match), answered by a later WAKE |
| request to the victim line while its MSHR is MS_WB/MS_REQ/MS_WAIT | `v_bits` already 0: a new miss, another way (`way_claimed` excludes the first) |
| memory response arrives while the engine is on a hit | MS_FILL set immediately; at the next E_SEL the fill pre-empts new requests |
| `rsq_full` (4 responses not yet taken by the network) | E_SEL takes no new request; E_WAKE waits; hits already past E_SEL still push |

### 3.7 DCO features on the board configuration

At reset policy none changes cycle behaviour. If a program sets `policy`
(TMU offset 0, `tnca_tmu.v:232`): `pol_repl != 0` with no invalid way adds
E_RANK (RANK_PIPE = 1, 542-543, 1233-1239, 1248-1257); `pol_dbp` enables
tile-last-word notices (`tl_v`, one-entry buffer, 1480-1489; a second notice
while one waits is dropped, no stall); `pol_dyn`/`pol_tbyp` can bypass read
misses (MSHR without a way: fill writes no array, WAKE reads `ms_line`,
1450-1451; stores to a bypassed line block, 1262-1266). TMU register stores
wait `tmu_wr_ready` (862, 1137): round robin among slices, one per cycle,
plus one cycle of capture with REGISTER_WRITES = 1 (`tnca_tmu.v:190-215`,
323-330). TMU loads are combinational (229-242).

### 3.8 Maintenance (flush) path (board only, MAINTENANCE = 1)

`maint_ready` (573-574): `est == E_SEL && rq_v == 0 && rsq_cnt == 0 && tg_v == 0
&& all MSHRs free && !mem_rsp_valid`. Then per line index 0..SETS*WAYS-1:
E_MTAG (tag read) -> E_MSCAN: valid, overlapping and dirty -> E_MREAD ->
E_MSEND (waits `mem_req_ready`) -> next; else invalidate if overlapping ->
next (1085-1118). Cost: 2 cycles per clean/invalid line, 4+ per dirty line;
256 lines on the board, so >= 512 cycles per flush per slice, slices in
parallel. `maint_done` after the last index; `tnca_cache_handoff` (`rtl/nuca/
tnca_cache_handoff.v:14-26`) accepts only when `quiescent` (`tnca_top.v:219,
230`: all tiles idle, no dispatch busy, home links idle, no host op, no
launch) and reports `done` when every home is done and `memory_drained`
(`tnca_multicore_axi.v:109`: AXI adapters and links idle). Requests arriving
during maintenance are refused at `req_ready` (393). It never runs during a
launch.

## 4. Home to outlet, and the outlet

### 4.1 Home link (`tnca_top.v:196-216`, `rtl/common/tnca_line_link.v`)

MEM_HOME_STAGES = 0 (sim): wires (198-205). MEM_HOME_STAGES = 2 (board):
request side STAGES x `tnca_fifo #(.DEPTH(2))` (37-39): +2 cycles minimum,
`m_req_ready` seen by the slice = stage-0 FIFO `cnt < 2`, so the slice's E_MREQ
usually fires immediately and the stall of the memory side is absorbed by 4
line entries; response side STAGES registers, no back-pressure (40-46): +2
cycles. `idle` = all stages empty and no reply in flight (34, 46); it gates
maintenance and `cache_quiescent`.

### 4.2 Memory arbiter (`rtl/nuca/tnca_mem_arbiter.v`), PORTS = 1

| rule | lines |
|---|---|
| `pick`: scan `(rr + ai) % N` for ai = N-1..0; last valid wins, i.e. the first valid at or after `rr` | 60-68 |
| `held`: on `m_req_valid && !m_req_ready` latch `held_pick`; while held, `pick = held_pick`, `pick_ok = s_req_valid[held_pick]` | 69-71, 94 |
| handshake: `held <= 0`, `rr <= pick + 1 (mod N)` | 95-98 |
| `s_req_ready[g] = pick_ok && pick == g && m_req_ready` (combinational) | 84 |
| response: `s_rsp_valid[g] = m_rsp_valid && m_rsp_src == g` (combinational) | 85-87 |
| `m_req_src = pick`, `m_req_tag = slice's MSHR index` | 78-79 |

Zero cycles in either direction. Three homes competing for one DRAM port each
get one grant per cycle in rotation, subject to `m_req_ready` (sim:
`lfsr[5]`, sec 5). The DRAM's `rsp_src`/`rsp_tag` come back untouched; the
arbiter steers by `src`, the slice indexes `ms_*` by `tag`.

### 4.3 PORTS = 3 (board): `tnca_mem_ports.v`, `tnca_memory_mode.v`, `tnca_line_axi.v`

| stage | rule | lines |
|---|---|---|
| `tnca_mem_ports` | home `s` -> port `s % PORTS` (= `s` for N = PORTS = 3); one `tnca_mem_arbiter` per port with a single requester; `legal_source = src < N && src % PORTS == p`; combinational | 37-56 |
| `tnca_memory_mode` (MEMORY_CONTROLS = 1) | `single_port = (memory_active_ports == 1)`: one `tnca_mem_arbiter` (round robin + hold) merges all ports onto port 0, responses of port 0 fanned out by `src % PORTS`; otherwise straight through; combinational | 29-54 |
| `memory_active_ports`, `memory_outstanding` | MMIO register 0x110, reset to PORTS and AXI_OUT_N (= 8); runtime sets them from `TNCA_RTL_MEM_PORTS` / `TNCA_RTL_OUTSTANDING` (`tnca_multicore_mmio.v:105,114`; `run_graphs.py:45-46` uses 1 / 1 for recordings) | |
| outlet link | `tnca_line_link` STAGES = LINK_STAGES = 2: +2 request, +2 response, 4 request entries | `tnca_multicore_axi.v:279-286` |
| `tnca_line_axi` admission | `req_ready = !fault && !issue_valid && free_slot >= 0 && !hazard && legal_address && admission_ok`; `admission_ok = active_count < admission_limit` (DYNAMIC_LIMIT = MEMORY_CONTROLS); `hazard` = an active slot on the same line where either is a write; OUT_N = 8 slots; a slot stays `active` until the last read beat or the write response B | 61-75, 114-152 |
| read issue | accept at `a`; AR valid during `a+1`; on `arready` `issue_valid <= 0`; next accept during `a+2`: 2 cycles per read | 77-80, 118-122 |
| write issue | accept at `a`; AW + W beat 0 during `a+1`, W beat 1 (`wlast`) during `a+2`; next accept during `a+3`: 3 cycles per write (BEATS = 512/256 = 2) | 81-86, 123-132 |
| read return | R beats accumulate into `line[rid]`; `read_final` on the last beat; `rsp_valid <= 1` at that edge: `rsp_valid` one cycle after the last beat, with `src`/`tag` of the slot | 87-88, 133-144 |
| write completion | B releases the slot; NO response to the home (WRITE_RESPONSES = 0) | 145-152 |
| `m_rready = 1` always; `m_bready = !read_final` (one completion per cycle) | 90-92 |

Board path from the slice's E_MREQ fire at `E_m`: home link +2, ports/mode 0,
outlet link +2 -> adapter accept during `m+4` at the earliest; AR during
`m+5`. From the last R beat during `q-1`: adapter `rsp_valid` `q`, outlet link
+2, mode/ports 0, home link +2 -> slice `mem_rsp_valid` during `q+4`. With
`TNCA_RTL_OUTSTANDING = 1` (recordings) the adapter admits ONE line at a time
(reads and writes alike), so a writeback blocks the following read until B.

Reordering: AXI reads may return out of order (independent IDs = slot
indices); each response carries its own `src`/`tag`, nothing in the slice
assumes order. The DRAM model reorders deliberately (sec 5).

## 5. `tb/common/tnca_dram_model.v` exactly

Instantiated by the bench (device.v:53) with `TAG_W 3`, `SRC_W 2`,
`LINES = TOTAL_WORDS/8`, `LATENCY 11`, `RANDOM_STALL 1`, `REORDER 1`,
`INTERVAL 1` (default), `SEED 32'hACE1_2345` (default), `POOL 32`.

| item | rule | lines |
|---|---|---|
| LFSR | `lfsr <= SEED` at every posedge with `rst_n = 0`; afterwards every posedge `lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]}`, independent of traffic | 52-56 |
| `req_ready` | `(RANDOM_STALL ? lfsr[5] : 1) && gap == 0`; INTERVAL = 1 keeps `gap = 0` always | 61-68 |
| accept (read) at `E_a` | slot = lowest free pool index; `p_data <= mem[line]` (old value, 0 if out of range); `p_tag`, `p_src`; `p_cnt <= LATENCY-1 = 10` | 90-106 |
| accept (write) | `mem[line] <= wdata` at `E_a`; no pool entry, no response ever | 97-99 |
| countdown | every posedge, every valid entry with `p_cnt != 0` decrements (pre-edge values; the entry accepted at this edge is not decremented) | 109-110 |
| eligibility | `p_valid && p_cnt == 0` evaluated with pre-edge values: an entry accepted at `E_a` is eligible at `E_{a+LATENCY}` | 112-114 |
| response | one per cycle: `rsp_valid <= 1` at that edge, so `rsp_valid` during `a+LATENCY+1 = a+12` at the earliest; `rsp_valid` is a 1-cycle pulse (`rsp_valid <= 0` default) | 88, 116-131 |
| pick among eligible | `pick = REORDER ? lfsr[9:0] % nelig : 0`; the `pick`-th eligible entry in ascending pool index | 117-129 |
| pool overflow | a read accepted with no free slot is silently DROPPED (`else if (slot >= 0)`); unreachable with <= 24 outstanding reads (3 slices x 8 MSHRs) | 100 |

What is random: only `lfsr`, a deterministic function of the number of posedges
since `rst_n` rose (the bench raises `rst` after 5 negedges, device.v:108;
the trace's `cycle` counts posedges with `rst_n` high from 0,
`tools/performance_harness/align/tnca_core_trace.v:22`). A model that advances
the same 32-bit LFSR with the same seed from the same origin predicts
`req_ready` in every cycle and every reorder pick exactly, PROVIDED it also
reproduces the exact accept cycle of every request (which fixes the pool
indices and the eligible set). It cannot predict them from relative timing
alone: the launch's absolute cycle depends on host mailbox polling
(device.v:111-113, `pollcycles` negedges per poll) and on how many host words
preceded it, so two runs of the same program see different stall patterns
unless the absolute cycle is known. Replaying the recorded `M`/`D` events
(`tnca_core_trace.v:90-94`, accept and response cycles per tile) sidesteps
this entirely. With `nelig > 1` the responses of equally-old reads serialise
one per cycle in LFSR-chosen order, so a burst of 3 misses to 3 slices returns
over 3 consecutive cycles in a permutation.

Write response: none. The slice does not wait for one (E_WBSND -> MS_REQ on
the request handshake, 1374-1379).

## 6. Launch, dispatch, completion

### 6.1 Shell (`rtl/tnca_multicore_shell.v`)

`cmd_ready = rst_n && grid_ready && !code_busy && !bad_host_pending && !invalid_done`
(91); `grid_ready = dispatch_ready && !host_busy && !maintenance_requested &&
!maintenance_busy` (`tnca_top.v:286`); `dispatch_ready = launch_ready = !running
&& !done_valid` (`tnca_grid_dispatch.v:64`). A launch is accepted at `L` when
`cmd_valid && cmd_ready`; the dispatcher accepts the same cycle (`tnca_top.v:321`,
`launch_valid = grid_valid && ...`). `bad_pc`/`bad_chunk` (84-90) turn it into
`invalid_done` instead (167). `done_valid = invalid_done || grid_done` (93).

### 6.2 Dispatcher (`rtl/tnca_grid_dispatch.v`)

| at `E_L` (104-118) | `running <= 1`, `remaining <= groups_total[31:0]`, `active <= launch_cores`, `rr <= 0`, `core_busy <= 0`, `pending <= 0`, `x,y,z <= 0`, `done_cycles <= 0`; or `done_valid <= 1` immediately for a malformed or empty grid (110-115) |
|---|---|
| group rule | `requested_width` = 1/2/4 for `launch_group_log2` 0/1/2 (44-45); `groups_x = ceil(gx / width)`; `groups_total = groups_x * gy * gz` (46-47); each enqueue hands `(x, y, z)` to one core and advances `x += width`, wrapping to `y+1`, then `z+1` (147-151) |
| which core | each cycle while `running && remaining != 0`: the first `candidate = (rr + j) % CORE_N`, j = 0.., with `active && !core_busy && !pending` (74-86); at the edge: `pending[choice] <= 1`, `cmd_x/y/z[choice]`, `remaining--`, `rr <= choice + 1` (143-152). One core per cycle. |
| per-core args | `cmd_arg_base[a] = args_q + a * stride_q` by PHYSICAL core index (68-70), `cmd_pc = pc_q`, `cmd_nx/ny/nz = gx, gy, gz` (72) |
| `cmd_valid[c]` | `= pending[c] && running && !faulted && !fault_now` (76), combinational from the registered `pending` |
| accepted | `cmd_valid & cmd_ready` (58): `core_busy <= 1`, `pending <= 0` for that core, `issued += width` (61, 122-130) |
| completion | `core_done[c] && core_busy[c]`: `core_busy <= 0`, `completed += accepted_width` (61, 131); the core becomes a candidate again the cycle after |
| grid done | `(remaining == 0) && busy_after == 0 && pending_after == 0 -> running <= 0, done_valid <= 1` (153-155); `done_valid` cleared on `done_ready` (103) |
| `done_cycles` | +1 at every edge while `running` (120): = (cycle at whose edge `done_valid` is set) - L |
| abort/deadline | `tnca_grid_deadline` (`abort_valid` when `elapsed >= limit`, limit 0 = none), faults: `fault_now` (53-57), `cmd_cancel` (71), no fabricated completions |

### 6.3 Command and completion links (`tnca_top.v:290-319`)

| | DISPATCH_STAGES = 0 (sim) | DISPATCH_STAGES = 2 (board) |
|---|---|---|
| `cmd_ready[c]` | `= tile_idle[c]` (299): accepted only when the core is idle, and `tile_launch = cv && cr` the same cycle | `= in_ready` of the command link stage 0 (`tnca_elastic_link`), independent of the tile |
| `launch_valid` at core `c0` (first core) | `L+2` | `L+4` (`cv` after 2 stages; `cr = tile_idle && !started`, 305-306) |
| second, third core | `L+3`, `L+4` | `L+5`, `L+6` |
| re-launch of a core that went idle at `i` | `core_done` at `i` (`completion_valid = dispatch_busy && tile_idle`, 300) -> enqueue at `E_{i+1}` -> `cmd_valid` and launch at `i+2` (if `remaining != 0`) | `started && tile_idle` pushed at `E_i`, `completion_valid` at `i+2`, `core_done` at `i+2`, enqueue `E_{i+3}`, `cmd_valid` at `i+4`, launch at `i+6` |
| grid `done_valid` after the LAST core goes idle at `i` | `i+1` (`done_valid <= 1` at `E_i`) | `i+3` |
| `done_cycles` | `i - L` | `i + 2 - L` |

`tile_idle = core_idle` directly (`tnca_top.v:492`, `tnca_tile.v:227`);
`core_idle` falls the cycle after `launch_valid` (CORE_TIMING sec 6.3:
`running != 0`), so no spurious completion in the launch cycle (`core_busy`
is set at the same edge). The core's own launch timeline (first instruction in
X at launch+5) is CORE_TIMING sec 6.1.

`cmd_cores` with `TNCA_ACTIVE_CORES = 1` has one bit (`run_graphs.py:45`): only
that tile's core runs; all three slices still serve as homes.

### 6.4 Instruction memory

Per-core synchronous RAM, `instruction <= code[ia >> 3]` (`tnca_multicore_shell.v:130`):
one cycle, no uncore involvement. `CODE_LINK_STAGES` delays host code
uploads only (113-127), never fetches.

## 7. Host access during a launch

`host_allow = GLOBAL_GRID && HOST_ACCESS && dispatch_ready && !grid_valid &&
all_idle && !maintenance_requested` (`tnca_top.v:257`); `dispatch_ready =
!running && !done_valid`. So while a grid runs `host_allow = 0`, `offer_host =
0` (`tnca_idle_host_mux.v:21`), `use_host = host_busy`; `host_busy` can only be
set by `offer_host` or a host handshake (43-44), and a launch is not accepted
while `host_busy` (`grid_ready`, `tnca_top.v:286`; shell 194 also gates
`host_valid` with `(!cmd_valid || host_busy)`). Therefore during a launch the
mux is a wire for the core (`n_valid = c_valid`, `c_ready = n_ready`,
`c_rsp_valid = n_rsp_valid`): zero added cycles, no interference. The device
is idle during host transfers, as believed.

Indirect effect, which a model must reproduce: host words go through tile 0's
core NI to the HOME slice of each address (same hash) and leave state behind
(valid and dirty lines, `victim_ptr`, MSHR-free). A host upload before a launch
pre-fills the slices (sim bench: 512 B per slice, so mostly evicted; writes
of evicted dirty lines went to DRAM as writebacks); a download hits dirty
lines. Each host word is a full round trip (`host_busy` until `h_rsp_valid`,
36, 45), serialised by the bench's `host_word` task (device.v:73-84). With
CACHE_MAINTENANCE = 0 on the sim bench nothing is ever flushed between
launches. The memory arbiter's `rr` is part of that left-behind state: every
host miss is at least one port transaction from the home (the fill, a
writeback before it if the victim is dirty), granted one at a time since the
words are serialised, so `rr` resumes after the home of the last host miss
(`tnca_mem_arbiter.v`, sec 10). A model that warms the slices in zero time
must move the pointer the same way, or the first tie of the next launch is
granted in the wrong order (seen on one launch of one point).

## 8. Points I am not certain about (with where to look)

1. Whether any recorded program writes the TMU window (policy, tensor table,
   dead list). `compiler/python/tnca_cc/dco.py:485` emits `const(TMU_BASE)`;
   which compile knobs enable it is in `tools/performance_harness/systemc/
   run_graphs.py` (`knobs_for`, HARNESS). If it does, sec 3.7 applies
   (E_RANK, TLW notices, bypass) and `dco_owner_epoch` mismatches (slice
   1223, 1249, 1258-1260) add LOOK retries at commit cycles.
2. Whether the slice geometry of the recordings is really 4 x 2:
   `tb/device/tnca_multicore_device.v:37` is unconditional and
   `tools/performance_harness/align/record_rtl.py:55-72` uses that bench
   unchanged. Confirm with `bench/iverilog.cmd` of a run. If a board capture
   is ever aligned with the model's `sets=32`, the sim recordings cannot be.
3. The absolute-cycle origin of the DRAM LFSR vs the trace's `cycle`:
   `tnca_core_trace.v:22-28` counts posedges with `rst_n` high from 0;
   `tnca_dram_model.v:53-56` shifts at every posedge with `rst_n = 1`. I read
   the LFSR value during trace cycle k as SEED shifted k times (during cycle 0
   it is SEED itself), with `req_ready = lfsr[5]` of that value. Verify with
   one waveform before relying on replay-free prediction.
4. Icarus evaluation order between the slice's `always @(posedge)` NBAs that
   touch the same array in one edge: `rq_blocked <= 0` (1466) vs
   `rq_blocked[cur_rq] <= 1` cannot coincide (single engine), and `touch`'s
   NBAs to `age` only matter for `pol_repl != 0`; but `v_bits`/`d_bits`
   writes from E_MSCAN and E_FILL are exclusive by state. No conflict found;
   not simulated.
5. Dirty-writeback ordering at the DRAM model: a writeback and a later read of
   the same line from the same slice are serialised by the MSHR (MS_WB ->
   MS_REQ), and `mem[line] <= wdata` lands at the write's accept edge while the
   read captures `mem` at its own accept edge (`tnca_dram_model.v:98,102`), so
   the read sees the write. A writeback from slice A and a read from slice B
   cannot be the same line (unique home). OK by construction; stated, not
   simulated.
6. Router `rr` at reset favours port 0 (local) for every output (174); on 3x1
   the north/south candidates are always invalid, so the effective rotation
   is over {local, east, west} with the gaps costing nothing. Check that the
   model's round robin keeps the same relative order (local=0, east=2,
   west=4 in a 5-slot ring).
7. `E_WAKE` waiting on `rsq_full` (1411) and E_SEL refusing new requests on
   `rsq_full` (1069) couple the engine to the response network's local FIFO
   (4) and the downstream VN_RSP path. How often this triggers in the
   recordings is unknown; the model has no such coupling.
8. The board's AXI side (`tnca_line_axi.v`) is specified only up to its
   handshakes: AR/AW/W ready timing, R beat spacing, B latency and read
   reordering are HBM/interconnect properties outside the RTL
   (`r40-board-m2.cfg` fits them: `hbm_latency_cycles`,
   `tl_hbm_write_latency_cycles`, `tl_outlet_to_slice_cycles`).
9. Whether the recordings run with `memory_active_ports = 1`
   (`TNCA_RTL_MEM_PORTS = 1`, `run_graphs.py:45`) on the board: then all
   three homes share port 0 through a second `tnca_mem_arbiter` inside
   `tnca_memory_mode` (29-38), and with `memory_outstanding = 1` the adapter
   holds one line (read or write) at a time. The cfg says `active_outlets=3`,
   `outstanding=8`; `r40-board-m2.sources.json:12` says a record overrides.
   Confirm per record.
10. Maintenance (flush) between launches on the board: when the runtime issues
    `cache_maint_valid` (needs `python/tnca/backends/multicore_mmio.py`) and
    whether a recorded point's cycle count starts after the flush completes.
    Sec 3.8 gives the per-line costs; the sim bench never flushes.
11. `tnca_slice_ni.v:78` builds `req_tag` with a zero-width replication
    `{(TAG_W-5){1'b0}}` for TAG_W = 5; Icarus accepts it. Not a timing matter
    but it is the tag the slice echoes, so a tool that rejects it would change
    nothing on silicon.
13. The `S`/`R`/`M`/`D` uncore records cited from `tools/performance_harness/
    align/tnca_core_trace.v:90-94` exist only in the UNCOMMITTED working-tree
    copy (`git diff --stat` shows 160 changed lines there plus edits to
    record_rtl.py, record_model.py, tracediff.py, README.md, chip.cpp,
    cycle_core.inc, none made by this task); commit 1f82d54's monitor writes
    only `L`/`I`/`Q`/`P`/`E`. `record_rtl.py:55-72` was likewise read from the
    working tree. Everything under `rtl/` and `tb/` is unmodified at 1f82d54.
12. Core-side acceptance of a response in the same cycle as a request issue
    and whether `mem_rsp_valid` can coincide with a beat issue at the LSU: the
    core NI asserts `rsp_valid` unconditionally one cycle after the flit is
    at the router's local output (core NI 126), and the LSU handles responses
    in every state (CORE_TIMING sec 3.2). I did not re-read `tnca_lsu.v`
    352-378 for a same-cycle request/response corner.

## 9. Where the model-side description disagrees with the Verilog

Checklist against `tools/performance_harness/systemc/configs/r40-board-m2.cfg`
(`tl_*`, 81-95) and `include/tnca_timeline_memory.hpp`:

| key / model rule | model | Verilog | verdict |
|---|---|---|---|
| `tl_noc_inject_cycles` | 1 | core NI 0 + source router FIFO 1 | agrees as accounting (the 1 is the FIFO) |
| `tl_noc_eject_cycles` | 2 | `rsq` -> home router FIFO 1 + core NI register 1 | agrees |
| `noc_stages` (per hop) | 3 | board `1 + NOC_LINK_STAGES = 3`; sim bench `1` | board agrees; **sim disagrees** |
| `tl_slice_hit_occupancy` / `tl_slice_hit_latency` | 3 / 4 | SEL, LOOK, HIT; `rsp_valid` at accept+4 | agrees (when the engine is free at accept+1) |
| `tl_slice_miss_occupancy` | 4 (SEL/LOOK/MISS/ALLOC) | board 4; sim 3 (no E_ALLOC) | board agrees; **sim disagrees** |
| `tl_slice_mreq_occupancy` | 2 | SEL + MREQ, but MREQ repeats while `!mem_req_ready` (sim: DRAM stall 50% of cycles, arbiter rotation among 3 homes) | agrees only when ready; **engine stall not modelled** |
| `tl_slice_fill_occupancy` / `tl_slice_wake_occupancy` | 2 / 2 | SEL+FILL, SEL+WAKE per target | agrees |
| `tl_slice_writeback_occupancy` | 2 (E_WBRD/E_WBSND) | SEL + WBRD + WBSND = 3, plus `mem_req_ready` stall | **disagrees: 3** |
| `tl_slice_to_outlet_cycles` | 4 | board home link 2 + outlet link 2 (FIFO minimums); sim 0 | board agrees; **sim 0** |
| `tl_outlet_to_slice_cycles` | 13 fitted (RTL 5) | board: adapter register 1 + outlet link 2 + home link 2 = 5 from the last R beat; sim: DRAM `rsp_valid` at accept+12, no links | fit absorbs HBM; sim must use the DRAM rule |
| `tl_outlet_read_issue_cycles` / `write_issue` / `beat_cycles` | 2 / 3 / 1 | `tnca_line_axi` 2 / 3 / 2 beats (given AXI ready every cycle) | board agrees; sim: DRAM accepts 1/cycle when `lfsr[5]` |
| `tl_hbm_write_latency_cycles` | 23 fitted | board: slot held until B; sim: writes hold nothing and get no response | sim: no write occupancy at all |
| `outstanding` | 8 (cfg), record may override | board `memory_outstanding` register (1 in recordings); sim: POOL 32 reads, writes unlimited | **check per record**; sim effectively unlimited |
| `outlets` / `active_outlets` | 3 / 3 | board 3 ports, but `single_port` merges to 1 when `TNCA_RTL_MEM_PORTS = 1`; sim 1 | **check per record** |
| `sets` / `ways` / `mshrs` / `targets` | 32 / 8 / 8 / 16 | board same; sim **4 / 2** / 8 / 16 | **sim disagrees** |
| victim choice (`slice()` 140-146) | invalid-first lowest way, then `rr[s]` pointer skipping ways with a fill not yet `ready` | same, plus: `way_claimed` is any non-bypass MSHR in the set (incl. MS_WAKE until the last target), and a fully claimed set BLOCKS the request (retried when an MSHR retires) rather than waiting on a timeline | close; model's `set_busy` wait approximates blocking |
| merge (`slice()` 131-136) | target when the line has a fill not yet ready | Verilog merges whenever an MSHR exists for the line (even MS_WAKE with `v_bits = 1`), and serves targets strictly in order, each SEL+WAKE | agrees in spirit |
| request buffer | none (engine timeline only) | oldest-eligible arbitration, same-line shadowing, blocked retries only on MSHR retirement, 8 entries of back-pressure | **not modelled** |
| MSHR work pre-empting hits | none | at every E_SEL: WAKE > FILL > WB > REQ before any new request | **not modelled** |
| response queue coupling | none | `rsq_full` (4) stops selection and WAKE | not modelled |
| memory arbiter | none (per-outlet issue timeline) | round robin with hold among homes sharing a port; combinational | not modelled (matters on sim with 3 homes on 1 port, and on board with `single_port`) |
| DRAM model | n/a (board constants) | sim: `lfsr[5]` ready, LATENCY+1, one response per cycle, LFSR-ordered among eligible | **sim alignment needs the DRAM rule or replay** |
| host transfers (`access()`) | place the line untimed, `ready = 0`, `dirty` on write, rr advance | same placement rule and `victim_ptr` advance (slice 1341); but the real transfer also evicts (dirty -> DRAM) and the sim bench's 512 B slices evict almost everything | agrees on state transitions |
| dispatch (`dispatch_cycles=6`, `launch_cycles=5`) | 6 | board: `cmd` accepted at L -> core launch at L+4 (first core), done at i+3; sim: L+2, i+1 | **check which the 6 means**; sec 6.3 has the exact values |
