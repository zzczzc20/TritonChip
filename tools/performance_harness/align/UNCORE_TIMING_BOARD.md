<!-- Status: DRAFT, written 2026-10-27 by reading the Verilog at d292590 (rtl/ and tb/ unmodified in the
     working tree; python/tnca/backends/multicore_mmio_sim.py carries one uncommitted line, `harness=`,
     that does not change any parameter). Companion to UNCORE_TIMING.md: only what the board
     configuration ADDS or CHANGES is here; everything not mentioned is as in that document. Nothing
     below was simulated; section 8 lists what reading could not settle. -->

# TNCA uncore cycle timing, board configuration (draft, from the Verilog)

The *board configuration* is the board-top simulation bench
`tb/fpga/tnca_multicore_board_regs.v` as `MulticoreMMIOLink`
(`python/tnca/backends/multicore_mmio_sim.py:83-94`) and the recorder
(`tools/performance_harness/align/record_rtl.py:57-58`, `BOARD`) build it:

```
-s tnca_multicore_board_regs
-P N=3 PORTS=3 MEMORY_BYTES=1048576 STAGES=2 NOC_PIPE=2 DISPATCH_PIPE=2
   HOME_PIPE=2 CODE_PIPE=2 MAINTENANCE=1 MEMORY_CONTROLS=1
tb/common/tnca_axi_line_model.v   (the memory behind the three AXI ports)
```

The bench (line 42) instantiates `rtl/board/tnca_multicore_hbm_xo.v` (`dut`) ->
`rtl/fpga/tnca_multicore_fpga.v` (`device`, xo 337-372) ->
`rtl/fpga/tnca_multicore_axi.v` (`system`, fpga 99-141) ->
`rtl/tnca_multicore_shell.v` (`shell`, axi 115-151) -> `rtl/tnca_top.v`
(`u_system`, shell 171-178). Line numbers below are of the file named in each
table. Conventions are UNCORE_TIMING.md sec 0: cycle `t` ends at edge `E_t`;
"accepted at `t`" = `valid && ready` during `t`; `tnca_fifo` pushes/pops are
registered, a datum pushed at `E_t` is at the head during `t+1`, a full FIFO
refuses a push in the cycle it is popped.

## 0. Parameters: how the bench's values reach the modules

| bench parameter (`tnca_multicore_board_regs.v:5,42`) | xo parameter | reaches | value |
|---|---|---|---|
| `STAGES` | `LINK_STAGES` | `tnca_multicore_axi.v:279` outlet `tnca_line_link` per port | 2 |
| `NOC_PIPE` | `NOC_LINK_STAGES` | `tnca_top.v:541` -> `tnca_mesh.v:88-131` `tnca_elastic_link` per router-to-router link | 2 |
| `DISPATCH_PIPE` | `DISPATCH_STAGES` | `tnca_top.v:293,314` command and completion `tnca_elastic_link` per tile | 2 |
| `HOME_PIPE` | `MEM_HOME_STAGES` | `tnca_top.v:207` home `tnca_line_link` per slice; `tnca_top.v:237,244` maintenance links | 2 |
| `CODE_PIPE` | `CODE_LINK_STAGES` | `tnca_multicore_shell.v:113,125` code command/reply `tnca_elastic_link` per core | 2 |
| `PORTS` | `MEM_PORTS` | `tnca_top.v:592` `tnca_mem_ports`; one outlet link + adapter per port (`tnca_multicore_axi.v:276`) | 3 |
| `MEMORY_CONTROLS` | `MEMORY_CONTROLS` | `tnca_multicore_axi.v:258` `tnca_memory_mode`; adapter `DYNAMIC_LIMIT` (287); MMIO register 0x110 | 1 |
| `MAINTENANCE` | `CACHE_MAINTENANCE` | `tnca_top.v:227` `tnca_cache_handoff`; slice E_MTAG/E_MSCAN states | 1 |
| `MEMORY_BYTES` | (bench only) | line model `LINES = (131072+MEMORY_BYTES)/64 = 18432` (bench 149) | 1 MiB |
| (bench 42) `.SETS(4),.SET_W(2),.WAYS(2),.WAY_W(1)` | `SETS/SET_W/WAYS/WAY_W` | xo 368-371 -> fpga 132-135 -> axi 147-150 -> shell 178 -> `tnca_top.v:471` -> `tnca_tile` -> slice | 4 x 2 |

Fixed by the xo module itself (`tnca_multicore_hbm_xo.v:8-23`, not
overridable from the bench): `AXI_DATA_W = 256`, `AXI_OUT_N = 8`, `AXI_ID_W = 3`,
`CODE_WORDS = 16384` (so `CODE_BYTES = MEM_BASE = 0x20000`), NX x NY = 3 x 1
(xo 337), `REG_ADDR_W = 9` (the 0x110/0x118 memory-control registers need 9
bits, `tnca_multicore_mmio.v:86`), `MEM_BASE = 0x20000` (fpga 22).

**Where the bench differs from the U280 image, by reading the parameter lists.**
`evidence/r40/v2/convergence/attempt03/image.tcl:1` sets `SETS 32 SET_W 5 WAYS 8
WAY_W 3` and ALSO `NUCA_REGISTER_VICTIM_SELECT 1`, `NUCA_CACHE_DATA_PIPELINE 1`,
`NUCA_REGISTER_VICTIM_RANK 1`, `DCO_LOCAL_REPLICAS 1`, `DCO_REGISTERED_RANGES 1`,
`DCO_REGISTER_WRITES 1`, `DCO_TLW_PIPE_STAGES 2`. The xo defaults for those seven
are 0 (`tnca_multicore_hbm_xo.v:8-14`) and the bench passes none of them (bench
42), so the board-top BENCH runs the slice without E_ALLOC (3-cycle miss
allocation, as the sim bench, UNCORE_TIMING sec 3.3), without the data-array
pipeline and with the unpipelined shared TMU. Sets/ways are therefore NOT the
only slice-level difference between this bench and the image: the image's slice
is the "board" column of UNCORE_TIMING sec 3, this bench's slice is the "sim"
column with the same 4 x 2 geometry. Every other image parameter (links,
dispatch, ports, memory controls, maintenance, AXI geometry, core features)
equals this bench's. The model that aligns against this bench therefore keeps
the sim-bench slice unchanged and changes only what sections 1-6 describe.

Reset: the bench holds `rst_n_async` low for 8 posedges and releases it `#1`
after the eighth (bench 330-331); the xo synchronises it with two flops (xo
131-135), so the device's `rst_n` rises two posedges later. The line models and
the AXI-Lite bridge use the asynchronous reset directly (bench 150, xo 138);
the device's `rst_n` is the synchronised one. Clock: one 10 ns clock for
everything (bench 7, xo 43); the AXI-Lite bridge's two clock domains are the
same clock here.

## 1. Pipeline and link stages

### 1.1 The two primitives

`tnca_elastic_link` (`rtl/common/tnca_elastic_link.v`): STAGES chained
`tnca_fifo #(.DEPTH(2))` (19-25); STAGES = 0 is a wire, `in_ready = out_ready`
and `out_valid = in_valid` combinationally (12-17). Each stage: `in_ready =
count < 2` of THAT stage only (fifo 32); a push at `E_t` is visible at the
stage's output during `t+1`. Latency when empty: exactly STAGES cycles from
`in_valid` at `t` to `out_valid` at `t+STAGES` (the head moves one stage per
edge because the next stage's `in_ready` is 1 while it holds < 2). Capacity:
2 x STAGES items. Back-pressure: when `out_ready = 0`, stage STAGES-1 fills to 2,
then stage STAGES-2, ..., then `in_ready` falls; the sender sees the stall only
after 2 x STAGES items have been absorbed, and the first item leaves 1 cycle
after `out_ready` returns (pop registered). No pop-through: a full stage being
popped at `E_t` still refuses a push during `t`.

`tnca_line_link` (`rtl/common/tnca_line_link.v`): the request side is the
same chain of `tnca_fifo #(.DEPTH(2))` over the 582-bit line request (37-39,
`QW = 64+1+512+SRC_W+TAG_W`); the response side is STAGES plain registers
(`rv <= reply_valid[s+1]`, 41-45) with NO ready: one line per cycle, never
stalled, +STAGES cycles. `idle` = every request stage empty and no reply
register valid and `!m_rsp_valid` (26, 34, 46).

### 1.2 Where each parameter sits

| parameter | path | module, lines | request/forward latency | reverse latency | capacity | `ready` seen upstream |
|---|---|---|---|---|---|---|
| NOC_LINK_STAGES = 2 | router -> router, per direction, per VN (east/west only on 3 x 1) | `tnca_mesh.v:88-91,103-106` `tnca_elastic_link #(.W(220))` | per hop `1 + 2 = 3` cycles head-to-head (router FIFO 1 + 2 stages) | n/a (each direction has its own link) | 4 (router input FIFO) + 2 x 2 per hop per VN | `r_out_ready[EC]` = stage-0 `count < 2`; independent of the far router |
| DISPATCH_STAGES = 2 | dispatcher `cmd_valid[c]` -> tile `launch_valid` | `tnca_top.v:293-296` `commands`, W = COMMAND_W (`128+192+NUM_RCTX+4+NUM_RCTX*PHYS_W`) | +2 (push `E_a`, `cv` at `a+2`) | completion: `started && tile_idle` -> `core_done` +2 (`completions`, 314-316, `out_ready = 1`) | 2 x 2 each way | `cmd_ready[c]` = stage-0 `count < 2`, independent of `tile_idle` (STAGES = 0: `cmd_ready = tile_idle`, 299) |
| MEM_HOME_STAGES = 2 | slice `mem_req_*` -> `tnca_mem_ports` input (`outlet_req_*`) and back | `tnca_top.v:207-214` `tnca_line_link` per home | +2 request | +2 response (registers, no back-pressure) | 2 x 2 requests | slice `mem_req_ready` = stage-0 `count < 2` |
| MEM_HOME_STAGES = 2 (maintenance) | `tnca_cache_handoff` -> slice flush command and `done` | `tnca_top.v:237-246` `tnca_elastic_link` W = 128 / W = 1 | +2 | +2 | 2 x 2 | only during a flush; never during a launch (UNCORE_TIMING sec 3.8) |
| LINK_STAGES = 2 | `tnca_memory_mode` port `p` -> `tnca_line_axi` adapter `p` and back | `tnca_multicore_axi.v:279-286` `tnca_line_link` per port | +2 request | +2 response | 2 x 2 requests | `mode_req_ready[p]` = stage-0 `count < 2` |
| CODE_LINK_STAGES = 2 | host code-store write/read: `code_fire` -> per-core RAM; RAM reply -> host | `tnca_multicore_shell.v:113-116` `commands` W = 137 (`out_ready = 1`), 125-127 `replies` W = 64 (`out_ready = 1`), 144-159 collector | +2 to the RAM write | +2 for the reply, +1 `reply_valid` register, +1 collector | one broadcast outstanding (122-124) | `code_command_ready` = stage-0 `count < 2` (never low: one outstanding) |

There are no other register slices on these paths in `tnca_top.v`: `roles_q`,
`slice_q`, `vbase_q` (282-285) are captured at the grid accept edge `E_L` and
are already stable when the first command is pushed at `E_{L+2}` (sec 5).
Instruction fetch (`instruction <= code[ia >> 3]`, shell 130) is one
synchronous RAM cycle and is untouched by `CODE_LINK_STAGES` (UNCORE_TIMING
sec 6.4).

### 1.3 STAGES = 2 versus 0, per path, in one line each

| path | STAGES = 0 (sim bench) | STAGES = 2 (board) |
|---|---|---|
| mesh hop, head at router A -> head at router B | 1 | 3; 4 extra flits of buffering per hop per VN before the upstream router's output stalls |
| dispatcher command -> `launch_valid` | same cycle as `cmd_valid`, and only if `tile_idle` | `cmd_valid` at `a` -> `launch_valid` at `a+2`; accepted by the link regardless of `tile_idle`, launched when `tile_idle && !started` |
| core idle -> dispatcher `core_done` | same cycle (`completion_valid = dispatch_busy && tile_idle`, 300) | `+2` |
| slice E_MREQ fire -> request at `tnca_mem_ports` | 0 (wires, 198-205) | +2; `mem_req_ready` is the link's, not the port's |
| port response -> slice `mem_rsp_valid` | 0 | +2 |
| `tnca_memory_mode` -> adapter `req_valid` | n/a (sim bench has no adapter) | +2; `mode_req_ready[p]` is the link's |
| adapter `rsp_valid` -> `tnca_memory_mode` | n/a | +2 |
| host code word: `code_fire` at `f` -> `host_rsp_valid` | `f+1` (shell 141-143) | `f+6` (sec 6) |

## 2. The memory side with MEM_PORTS = 3

### 2.1 Chain from a slice to its adapter

```
slice h  --mem_req/rsp-->  home link h (tnca_line_link, MEM_HOME_STAGES 2)      tnca_top.v:207
         --outlet_req[h]-> tnca_mem_ports (port p = h % 3 = h, combinational)      tnca_top.v:592, tnca_mem_ports.v:40-56
         --ports_req[p]--> tnca_memory_mode (combinational; single_port merges)   tnca_multicore_axi.v:259
         --mode_req[p]---> outlet link p (tnca_line_link, LINK_STAGES 2)          tnca_multicore_axi.v:279
         --qv/qr---------> tnca_line_axi adapter p                                 tnca_multicore_axi.v:287
         --m_axi_hbm{p}--> tnca_axi_line_model p (LATENCY 25 + 19 p)               bench 148-149
```

`tnca_top.v` with MEM_PORTS > 1 uses the `g_multiple_memory` branch (589-601):
the scalar `mem_req_*` port is tied to zero and `tnca_multicore_axi.v:230-232`
(`g_vector`) ties `mem_req_ready = 0`; nothing travels on it. `memory_reads /
memory_writes[p]` (tnca_top 555-561) count `ports_req` handshakes, i.e. after
the home link, before the mode block; they are what MMIO op 17 reports.

### 2.2 `tnca_mem_ports` (`rtl/nuca/tnca_mem_ports.v`)

| rule | lines |
|---|---|
| home `s` is bound to port `s % PORTS`; with N = PORTS = 3 port `p` serves exactly home `p` | 38-46 |
| one `tnca_mem_arbiter #(.N(3))` per port, whose `s_req_valid` is masked to that port's homes (`requests[s] = (s % PORTS == p) && s_req_valid[s]`) | 40, 48-56 |
| `s_req_ready[s] = pick_ok && pick == s && m_req_ready[p]` (arbiter 84): with one possible requester `pick` is always `s`, so `outlet_req_ready[h] = ports_req_ready[h]` combinationally, zero cycles | arbiter 60-72, 84 |
| `held`/`rr` (arbiter 69-71, 94-98) can only ever select home `p`: no hold rule has any effect; `m_req_src = pick = h` (79) | |
| response: `s_rsp_valid[h] = m_rsp_valid[p] && legal_source && m_rsp_src == h`, `legal_source = src < N && src % PORTS == p`; combinational | 37-38, 55, arbiter 85 |
| an illegal `src` on a port raises `protocol_fault[p]` (57-60) -> dispatcher abort (`tnca_top.v:321`) | |

Compared with PORTS = 1 (UNCORE_TIMING sec 4.2): no round robin among homes, no
`held`, no one-grant-per-cycle serialisation. The three homes issue and receive
in parallel on disjoint ports; home `h`'s only competitors for port `h` are its
own earlier requests (link FIFO order) and the adapter's admission rules
(sec 3). The latency a line sees is fixed by its home: `home = hash % 3`
(UNCORE_TIMING sec 2.2) -> port `home` -> model latency `25 + 19 * home`
(sec 4). A program whose lines hash to slice 2 waits 38 cycles more per miss
than one on slice 0.

### 2.3 `tnca_memory_mode` (`rtl/fpga/tnca_memory_mode.v`), MEMORY_CONTROLS = 1

`single_port = (memory_active_ports == 1)` (`tnca_multicore_axi.v:260`).
`memory_active_ports`/`memory_outstanding` reset to `PORTS = 3` / `AXI_OUT_N = 8`
(`tnca_multicore_mmio.v:105`); the host rewrites register 0x110 only when the
device is idle and drained (mmio 111-115). `MulticoreMMIOBackend.start`
(`python/tnca/backends/multicore_mmio.py:75-82`, 139) writes 0x110 at every
session start with `(physical_ports, maximum_outstanding) = (3, 8)` unless
`TNCA_RTL_MEM_PORTS` / `TNCA_RTL_OUTSTANDING` are set (multicore_mmio.py:24-27);
`record_rtl.py` sets neither (its `TNCA_RTL_*` pass-through, 199, is for core
flags). So the recordings run with `single_port = 0`, `admission_limit = 8`.

| mode | rule | lines |
|---|---|---|
| `single_port = 0` | pass-through: `s_req_ready[p] = m_req_ready[p]`, `m_req_valid[p] = s_req_valid[p]`, `s_rsp_valid[p] = m_rsp_valid[p] && legal_source`; zero cycles | 43-53 |
| `single_port = 1` | a `tnca_mem_arbiter #(.N(3))` over the three port requests (round robin + hold, UNCORE_TIMING sec 4.2) drives port 0 only; ports 1, 2 get `m_req_valid = 0`; port 0's response is fanned to `s_rsp_valid[p]` by `src % PORTS == p`; zero cycles both ways | 29-38, 41-53 |
| fault | a response on port `k` whose `src >= N`, or (single) `k != 0`, or (multi) `src % PORTS != k` | 56-62 |

In single-port mode all three homes share port 0's outlet link, adapter and
line model (LATENCY 25) and the arbiter's one-grant-per-cycle rotation returns.

## 3. `rtl/fpga/tnca_line_axi.v`: the adapter, one per port

Parameters as instantiated (`tnca_multicore_axi.v:287`): `SRC_W 2`, `TAG_W 3`,
`DATA_W 256`, `OUT_N 8`, `DYNAMIC_LIMIT = MEMORY_CONTROLS = 1`, `ID_W 3`,
`MEM_BASE 0x20000`, `WRITE_RESPONSES 0` (default). `BEATS = 512/256 = 2` (41).

### 3.1 State per slot `k` (0..7)

`active[k]`, `writing[k]`, `sent[k]`, `address[k]` (home address),
`bus_address[k] = axi_base + (address - MEM_BASE)`, `line[k]` (512 bits: the
write data, or the read data being assembled), `source[k]`, `tag[k]`,
`rbeat[k]` (43-48). One issue register set: `issue_valid`, `issue_slot`,
`aw_done`, `w_done`, `wbeat` (49-51). `active_count` = popcount of `active`
(57-60).

### 3.2 Admission (combinational, 61-75)

```
free_slot    = lowest k with !active[k], else -1                        (66-69)
hazard       = some active k with address[k] == req_addr && (writing[k] || req_we)   (70)
admission_ok = admission_limit >= 1 && admission_limit <= 8 && active_count < admission_limit   (61-62)
legal_address= req_addr >= MEM_BASE && req_addr[5:0] == 0 && axi_base[5:0] == 0 && axi_size >= 64
               && req_addr - MEM_BASE <= axi_size - 64 && axi_base <= 0xffff..ffc0 - (req_addr - MEM_BASE)   (63-65)
req_ready    = rst_n && !fault && !issue_valid && free_slot >= 0 && !hazard && legal_address && admission_ok   (75)
```

`admission_limit = memory_outstanding` (`tnca_multicore_axi.v:290`), 8 unless
the host lowered it; with 8 it is equivalent to `free_slot >= 0`. `axi_base`
and `axi_size` are MMIO registers 0xa8/0x100/0x108 and 0xb0
(`tnca_multicore_mmio.v:121-122`), which the host sets to the allocation
returned by `MulticoreMMIOLink.hbm_open` (`multicore_mmio_sim.py:44-48`:
base `0x1_0000_0000`, size <= 1 MiB). An illegal address is refused AND
latches `fault` (113), which stops admission for good (75) and aborts the grid
through `axi_fault -> platform_fault -> grid_abort` (`tnca_multicore_axi.v:105-106`,
shell 187).

What stalls `req_ready` during normal operation, in order of likelihood:
`issue_valid` (the previous request has not finished its AR, or AW + both W
beats: sec 3.3-3.4); `hazard` (a write to this line is still waiting for B, or
this is a write and any access to the line is still active); all 8 slots
active. Because the request sits at the head of the outlet link's last FIFO
stage, a stalled request blocks every request behind it on that port
(head-of-line; the link is a FIFO).

### 3.3 Accept and issue (104-132)

| event | edge / cycle |
|---|---|
| accept `req_valid && req_ready` during `a` | at `E_a`: `active[s] <= 1`, `writing[s] <= req_we`, `sent[s] <= 0`, `address`, `bus_address`, `line <= req_wdata`, `source`, `tag`, `rbeat <= 0`; `issue_slot <= s`, `issue_valid <= 1`, `aw_done <= 0`, `w_done <= 0`, `wbeat <= 0` (114-119) |
| read: `m_arvalid = issue_valid && !writing[issue_slot]` | high from `a+1` until `arready` (77); `araddr = bus_address[s]`, `arid = s`, `arlen = 1`, `arsize = 5` (32 B), `arburst = INCR` (78-80) |
| AR handshake during `x` (`x >= a+1`) | at `E_x`: `issue_valid <= 0`, `sent[s] <= 1`, `read_lines++` (120-122); `req_ready` may be high again during `x+1` |
| write: `m_awvalid = issue_valid && writing && !aw_done`; `m_wvalid = issue_valid && writing && !w_done` | both high from `a+1` (81, 84); `wdata = line[s][wbeat*256 +: 256]`, `wstrb` all ones, `wlast = (wbeat == 1)` (85-86) |
| AW handshake | `aw_done <= 1`, `write_lines++` (123); AW and W are independent, either may complete first |
| W handshake, beat 0 | `wbeat <= 1` (124-126) |
| W handshake, beat 1 (`wlast`) | `w_done <= 1` (126) |
| write issue complete: at the edge where `(aw_done || AW handshake now) && (w_done || (W handshake now && wlast))` | `issue_valid <= 0`, `sent[s] <= 1` (128-132); `req_ready` may be high again the cycle after |

Minimum occupancy of the issue register, AXI ready every cycle: read 2
cycles per request (accept at `a`, next accept at `a+2`); write 3 cycles
(accept `a`, AW + W0 during `a+1`, W1 during `a+2`, next accept `a+3`). Every
cycle of `arready = 0`, `awready = 0` or `wready = 0` adds one cycle to the
port's admission (sec 4.3 gives the model's ready probabilities).

### 3.4 Read return (87-90, 133-144)

`m_rready = rst_n` (90): every R beat is taken in the cycle it is offered.
`read_match = rid < 8 && active[rid] && !writing[rid] && sent[rid]` (87); a
beat without a match, with `rresp != 0`, or with `rlast` not on beat 1 latches
`fault` (134, 137). Each matching beat lands in `line[rid][rbeat*256 +: 256]`
and `rbeat[rid]++` (138, 142). On the beat with `rbeat == 1 && rlast`
(`read_final`, 88): `active[rid] <= 0`, `rsp_valid <= 1`, `rsp_src <= source[rid]`,
`rsp_tag <= tag[rid]`, `rsp_rdata <= line[rid]` with this beat patched in
(139-141). `rsp_valid` is a one-cycle pulse (`rsp_valid <= 0` every cycle,
112): **`rsp_valid` is high during the cycle after the last R beat is
accepted.** Beats of different IDs may interleave (one `rbeat` per slot);
there is one R channel, so at most one line completes per cycle.

### 3.5 Write completion (89, 92, 145-152)

`m_bready = rst_n && !read_final` (92): a B offered in the same cycle as a final
R beat waits one cycle (one completion per cycle at the home side, although
with `WRITE_RESPONSES = 0` the B produces no home response at all).
`write_match = bid < 8 && active[bid] && writing[bid] && sent[bid]` (89); on a
matching B: `active[bid] <= 0` (149); `bresp != 0` or no match latches `fault`.
**The slot stays `active` from the accept edge until the B handshake**, which
is what the hazard rule (sec 3.2) and `idle` (76: `!(|active) && !issue_valid`)
see. The home is never told about a write's completion (UNCORE_TIMING sec 4.3).

### 3.6 Accept-to-response with an AXI side that answers immediately

With `arready = 1` during `a+1`, R beats during `a+2` and `a+3` (the fastest an
AXI slave could answer): adapter `rsp_valid` during `a+4`; through the outlet
link (+2) and the home link (+2) the slice's `mem_rsp_valid` is during `a+8`.
Counting from the slice's E_MREQ fire at `m` (home link +2, ports/mode 0,
outlet link +2: adapter accept at `a = m+4` at the earliest, UNCORE_TIMING sec
4.3): slice response at `m+12`, the same number as the sim bench's DRAM model
(`LATENCY+1 = 12`, UNCORE_TIMING sec 1.2) but with the real AXI slave's
latency added on top (sec 4.4: the bench's line model makes it `m+37` on port
0). A write: accept `a`, AW+W0 `a+1`, W1 `a+2`, slot released by B (sec 4.4).

### 3.7 Ordering and hazards

| situation | behaviour |
|---|---|
| read after read, same line, both active | allowed to overlap (hazard needs a write on one side); the AXI slave may return them in either order; each carries its own `src`/`tag` |
| read after write (writeback then re-read of the victim line) | the read is refused at `req_ready` until the write's B (hazard); the slice's MSHR already serialises MS_WB -> MS_REQ, so the read's request reaches the adapter >= 1 cycle after the write's accept and then waits |
| write after read, same line | refused until the read's last beat |
| write after write, same line | refused until the first write's B |
| responses out of order | yes, by AXI ID = slot index; nothing in the adapter or the slice assumes order |
| writes vs reads to different lines | overlap freely up to 8 active slots; the only coupling is the single issue register (one AR or one AW/W burst in flight at the adapter's output at a time) |

## 4. `tb/common/tnca_axi_line_model.v`, as the bench instantiates it

Instantiated per port `p` (bench 148-179): `DATA_W 256`, `OUT_N 8`, `LINES 18432`,
`LATENCY 25 + 19 p` (25 / 44 / 63), `MODE 0` (default: OKAY responses),
`WRITE_DELAY 15` (default), `ID_W 3`. `memory[0..LINES-1]` is 512-bit lines
initialised to zero (18-19); line index = `(addr - 0x1_0000_0000) / 64` (47-48,
53-54); an address below `0x1_0000_0000`, unaligned, or past `LINES` is
`$fatal`, as are wrong burst geometry and an ID reused while active
(45-47, 52-53, 57, 61). All model state is updated with blocking assignments
inside one `always @(posedge clk)` (32-91); outputs are non-blocking. The
device-facing timing is therefore: a handshake during cycle `t` is processed
at `E_t`, and anything the model decides at `E_t` is visible during `t+1`.

### 4.1 Nondeterminism: exactly one source, seeded

`integer seed = 123` per instance (26); every random value is `$random(seed)`,
which advances `seed` in place. The stream is a deterministic function of the
number of calls, and the number of calls per posedge with `rst_n = 1` is

* 3, always: `m_awready <= cycle > 12 && (($random & 0x7fffffff) % 100) > 35`,
  `m_wready <= (... % 100) > 40`, `m_arready <= (... % 100) > 30` (86-88).
  Registered: the value drawn at `E_t` is the ready seen during `t+1`,
  whether or not anything is valid. Taking `% 100` as uniform: `awready` 64 %,
  `wready` 59 %, `arready` 69 % of cycles, independent per channel and per
  cycle. `awready` is forced low for the model's first 12 cycles.
* plus `max(0, n_due - 1)` for the read pick (72-74), where `n_due` is the
  number of reads with `reads[j] && due[j] <= cycle`: the loop
  `if (reads[j] && due[j] <= cycle && (pick < 0 || ($random(seed) & 1))) pick = j`
  draws only when the first two conjuncts hold and `pick >= 0`, under
  short-circuit evaluation (IEEE 1800-2012 11.3.5; the bench is compiled
  `-g2012`; not verified on Icarus, sec 8.3). The pick is evaluated every
  cycle because `!m_rvalid || m_rready` is always true (`m_rready = 1` from
  the adapter).

The three ports have three independent `seed` variables with the same initial
value, so their streams are identical and diverge only through consumption
(the `n_due - 1` draws). The model's `cycle` counter (25, 37) counts posedges
with `rst_n_async = 1`, which starts two posedges before the device's `rst_n`
(sec 0); it is never reset by a later reset (34-35 reset the queues, not
`cycle` or `seed`). No other `$random`, `$urandom` or time-dependent
construct exists in the RTL or the bench; the remaining variability is the
host's wall-clock arrival of mailbox batches (bench 344-358, polled every
`pollcycles = 20` posedges), which fixes the absolute cycle of each launch and
therefore which draws a launch meets. The bench's `+journal=` / `+replay=`
(`tb/fpga/tnca_mmio_journal.vh`, `multicore_mmio_sim.py:39`) exist to replay
that arrival timing.

### 4.2 Reads: AR -> R

| step | rule | lines |
|---|---|---|
| AR handshake during `x` | at `E_x` (with `cycle = C_x` after the increment at 37): `reads[id] = 1`, `location`, `beat = 0`, `due[id] = C_x + LATENCY + (id * 11) % 31` | 44-50 |
| eligible | `reads[j] && due[j] <= cycle`: first true at `E_{x + LATENCY + skew}` | 74 |
| pick, every edge (since `m_rready = 1`) | the lowest due index, then each higher due index replaces it with probability 1/2 (ascending scan): with `n` due reads P(highest) = 1/2, next 1/4, ..., the two lowest 1/2^(n-1) each | 72-74 |
| beat out | `m_rvalid <= 1`, `m_rid`, `m_rdata = memory[location][beat*256 +: 256]`, `m_rlast = (beat == 1)`, `rresp = 0` | 75-79 |
| beat accepted during `y` | at `E_y`: `beat[rid]++`, or `reads[rid] = 0` on `rlast`; a fresh pick follows in the same block | 65-67 |

Per-slot skew `(id * 11) % 31` for slots 0..7: 0, 11, 22, 2, 13, 24, 4, 15.
Timeline for one read alone, AR handshake during `x`, slot `s`:
first beat valid during `x + LATENCY + skew(s) + 1`, second beat during
`x + LATENCY + skew(s) + 2` (the same ID is picked again when it is the only
due read), adapter `rsp_valid` during `x + LATENCY + skew(s) + 3`. With two or
more due reads the beats interleave at random, one beat per cycle in total, so
a line's second beat waits behind the others' beats with the probabilities
above: with `n` lines due simultaneously, all `2n` beats leave over `2n`
consecutive cycles in a random interleaving. Reads therefore reorder for two
reasons: the deterministic per-slot skew (slot 1 is 11 cycles later than slot
0, slot 3 only 2) and the random pick among due reads.

### 4.3 Writes: AW / W -> B

| step | rule | lines |
|---|---|---|
| AW handshake | `aw_have = 1`, `aw_slot = awid`, `aw_line`; a second AW while `aw_have` is `$fatal` (one write burst in flight per port, which the adapter guarantees through its single issue register) | 51-55 |
| W handshake | `incoming[wcount*256 +: 256] = wdata`, `wcount++`; `wlast` must be on the second beat | 56-59 |
| commit, at the edge `E_w` where (after the same-edge AW/W updates) `aw_have && wcount == 2` | `writes[slot] = 1`, `write_data`, `due[slot] = C_w + LATENCY + WRITE_DELAY + (slot * 7) % 23`, `aw_have = 0`, `wcount = 0` | 60-64 |
| B pick, every edge with `!m_bvalid || m_bready` | the LOWEST due write index (descending scan, last assignment wins); `m_bvalid <= 1`, `m_bid`, `bresp = 0` | 81-84 |
| B handshake during `b` | at `E_b`: `memory[location] = write_data` (the write becomes visible to reads only now), `writes[bid] = 0` | 68-70 |

Per-slot skew `(id * 7) % 23` for slots 0..7: 0, 7, 14, 21, 5, 12, 19, 3. With
AW and both W beats accepted by `E_w`, `bvalid` is first high during
`w + LATENCY + 15 + skew'(s) + 1`; the adapter takes it that cycle unless a
final R beat coincides (`bready = !read_final`, then the B holds: no re-pick
while `m_bvalid && !m_bready`). The model counts `collisions` (71) but does
nothing with them.

### 4.4 End to end on this bench, port `p`, slot `s`, no stalls

Let the slice's E_MREQ fire at `m` (`mem_req_valid && mem_req_ready`), the
adapter accept `a = m+4`, and every AXI ready be high when needed.

| event | cycle | port 0, slot 0 |
|---|---|---|
| adapter accept | `a = m+4` | `m+4` |
| AR handshake | `x = a+1` | `m+5` |
| first R beat | `x + L_p + skew(s) + 1` | `m+31` |
| second R beat | `+1` | `m+32` |
| adapter `rsp_valid` | `x + L_p + skew(s) + 3 = m + L_p + skew(s) + 8` | `m+33` |
| outlet link `s_rsp_valid` | `+2` | `m+35` |
| slice `mem_rsp_valid` | `m + L_p + skew(s) + 12` | **`m+37`** (port 1: `m+56`, port 2: `m+75`) |
| slice `rsp_valid` to the network (SEL, FILL, SEL, WAKE) | `+5` | `m+42` |

A writeback (dirty victim) accepted at `a`: AW + W0 during `a+1`, W1 during
`a+2` (`E_w = E_{a+2}`), B during `a + L_p + 15 + skew'(s) + 3`; the slot is
`active` for `a+1 .. a + L_p + skew'(s) + 18` = 43 cycles on port 0 (62 on port
1, 81 on port 2) at slot 0. During that time a read of the same line is
refused (hazard) and the port's `idle` is low (sec 6: completion gating).

Each `arready = 0` cycle before the AR handshake, each `awready = 0` /
`wready = 0` cycle during a write issue, delays that request AND holds the
adapter's `req_ready` low for the request behind it. With the model's ready
rates the expected read issue cost is `1/0.69 = 1.45` cycles of AR plus the
accept cycle; a write needs an AW handshake (expected 1.56 cycles) and two W
handshakes (expected 1.69 each), overlapped where the random readies allow.

## 5. Dispatch and completion with DISPATCH_STAGES = 2

Everything in UNCORE_TIMING sec 6.1-6.2 holds (shell `cmd_ready`, dispatcher
rules). The board adds, in `tnca_multicore_axi.v`: the shell sees
`cmd_valid && memory_drained` (157) and the MMIO sees `cmd_ready =
shell_cmd_ready && memory_drained` (110), `done_valid = shell_done_valid &&
memory_drained && shell_all_idle` (111), `all_idle = shell_all_idle &&
memory_drained` (113), with `memory_drained = (&axi_idle) && (&link_idle)`
(109): every adapter has no active slot and nothing issuing, every outlet
link is empty. The shell's `done_ready` is likewise gated (173).

### 5.1 Launch to the first core, derived (`tnca_grid_dispatch.v`, `tnca_top.v:290-319`)

Let `L` be the cycle in which `launch_valid && launch_ready` at the dispatcher
(= the shell's `cmd_valid && cmd_ready`, same cycle).

| cycle | dispatcher (`tnca_grid_dispatch.v`) | command link (`tnca_top.v:293-296`) | tile |
|---|---|---|---|
| `E_L` | `running <= 1`, `remaining <= groups_total`, `active <= launch_cores`, `rr <= 0`, `core_busy <= 0`, `pending <= 0` (104-118) | | |
| `L+1` | `enqueue[c0]` for the first `candidate = (rr + j) % N` with `active && !core_busy && !pending` (74-86); at `E_{L+1}`: `pending[c0] <= 1`, `cmd_x/y/z[c0]`, `remaining--`, `rr <= c0 + 1` (143-152) | | |
| `L+2` | `cmd_valid[c0] = pending[c0] && running` (76); `cmd_ready[c0]` = stage-0 `count < 2` = 1; `accepted[c0]`: at `E_{L+2}` `core_busy[c0] <= 1`, `pending[c0] <= 0`, `issued += width` (58, 61, 122-130). Also `enqueue[c1]` (rr = c0+1): `pending[c1] <= 1` | push at `E_{L+2}` | |
| `L+3` | `cmd_valid[c1]`, accepted; `enqueue[c2]` | c0's command at the head of stage 0; moves to stage 1 at `E_{L+3}` | |
| `L+4` | `cmd_valid[c2]`, accepted | `cv[c0] = 1` (stage-1 head); `cr = tile_idle[c0] && !started` = 1 (305) | **`tile_launch[c0]` = `launch_valid` during `L+4`** (306); `started <= 1` at `E_{L+4}` (310) |
| `L+5`, `L+6` | | | `launch_valid` for c1, c2 |

With `TNCA_ACTIVE_CORES = 1` (the recordings, UNCORE_TIMING sec 6.3) only one
core is in `launch_cores`, launched at `L+4`. STAGES = 0: `cv = command_valid`
and `cr = tile_idle` are wires (298-299), launch at `L+2`, `L+3`, `L+4`.
`core_idle` (`tnca_core.v:2729`, `running == 0`) falls at `E_{L+4}`, so during
`L+5` `tile_idle = 0` and `started = 1`: no completion is pushed in the launch
cycle.

### 5.2 Completion and re-launch, derived

Let the core's `core_idle` (= `tile_idle[c]`, `tnca_tile.v:227`,
`tnca_top.v:492`) rise during `i`.

| cycle | completion link (`tnca_top.v:307-317`) | dispatcher |
|---|---|---|
| `i` | `in_valid = started && tile_idle = 1`; `ready` (stage-0 `count < 2`) = 1: push at `E_i`, `started <= 0` (311) | |
| `i+1` | head of stage 0 | |
| `i+2` | `completion_valid[c]` (stage-1 head, `out_ready = 1`) = `core_done[c]` | `core_done && core_busy`: at `E_{i+2}` `core_busy[c] <= 0` (`busy_after`, 61, 122), `completed += accepted_width` (131) |
| `i+3` | | `candidate` requires `!core_busy`: now true -> `enqueue[c]` if `remaining != 0` (and `c` is the first candidate from `rr`); `pending[c] <= 1` at `E_{i+3}`. If this was the last group and no core is busy or pending: `running <= 0`, `done_valid <= 1` at `E_{i+2}` already (153-155) -> **`done_valid` during `i+3`** |
| `i+4` | `cmd_valid[c]`, accepted, push at `E_{i+4}` | |
| `i+6` | `cv = 1`, `cr = tile_idle && !started = 1` | **re-launch during `i+6`** |

`done_cycles` increments at every edge from `E_{L+1}` to `E_{i+2}` inclusive
(120: `running` is still 1 at `E_{i+2}`), so `done_cycles = i + 2 - L`. STAGES =
0: `core_done` during `i`, `done_valid` during `i+1`, re-launch at `i+2`,
`done_cycles = i - L`. **The 'board' column of UNCORE_TIMING sec 6.3 (`L+4`,
`L+5`, `L+6`; re-launch `i+6`; `done_valid` `i+3`; `done_cycles = i+2-L`) is
confirmed by this reading.**

### 5.3 What the host sees as done

`grid_done` (`tnca_top` `grid_done_valid`) is the dispatcher's `done_valid`
during `i+3`. The MMIO's `done_valid` (`tnca_multicore_axi.v:111`) is that AND
`shell_all_idle` (= all tiles idle, no `dispatch_busy`, home links idle,
`!code_busy`: shell 71, `tnca_top.v:219-226`) AND `memory_drained`. During
`i+3` the tiles are idle, `dispatch_busy = 0` (cleared at `E_{i+2}`) and the
home links are empty (the last response left before the core could finish),
so the only thing that can delay the host-visible completion is an adapter
still holding a slot: a writeback waiting for its B (sec 4.4: B during
`w + L_p + 16 + skew'` for a last W beat during `w`; 43 / 62 / 81 cycles of
slot occupancy from the accept at slot 0) or a read in flight (impossible:
the core waited for it). The MMIO
(`tnca_multicore_mmio.v:189-193`) then latches `rsp_seq <= cmd_id`,
`rsp_r0/r1 <= done_cycles`, `launch_end <= device_clock` at the end of the
first cycle with `done_valid`, and the dispatcher clears `done_valid` at the
same edge (`done_ready = state == WAIT`, mmio 58, gated by `memory_drained &&
shell_all_idle`, axi 173). The cycle count the host reports (`R_RSP_R0/R1`,
`python/tnca/backends/u50.py:114-115, 557-558`) is `done_cycles`, unaffected by
the drain; `launch_end - launch_start` (registers 0xe8/0xf0, read by
`launch_timestamps`, multicore_mmio.py:160-161) is `done_cycles + 2` when
nothing is draining (sec 6) and larger by the drain wait otherwise.

The NEXT launch is additionally held by `cmd_ready = shell_cmd_ready &&
memory_drained` (axi 110) and by the doorbell's own `all_idle` condition (mmio
130); in practice the host's register traffic between launches (sec 6) is far
longer than any drain.

## 6. Host paths that differ, as far as a launch is concerned

**Register path.** The bench drives the xo's AXI-Lite slave on the same clock
(bench 211-231, xo 137-157, `ADDR_BITS 9`); `tnca_axi_lite.v` is a toggle
handshake across what is here one clock. For a 64-bit register the host writes
two 32-bit halves (bench 247-248); the low half is only latched
(`shadow_lo`, 250), the high half performs the write. Let `E_0` be the edge at
which the bridge accepts the high half's AW and W (both during the same cycle,
201-202, 223-230): `E_1` `aw_have && w_have` -> `wr_tog` toggles (237-245);
`E_2..E_4` `wr_sync` shifts (408); `wr_pulse = wr_sync[2] ^ wr_sync[1]` is high
during cycle 4 (369); at `E_4` `reg_we_q <= 1`, `wr_addr_q`, `reg_wdata_q`,
`wr_ack_tog` toggles (416-421): **`reg_we` is high during cycle 5 and the
register is written at `E_5`** (443-445, mmio 110). The acknowledgement returns
through `wr_ack_sync` (219): `bvalid` during cycle 8 (204, 261-267); the bench
takes it at `E_8` and starts its next bus write at the negedge after `E_9`
(217-218, 211), so the next AW/W is accepted at `E_10`: 10 cycles per high
half, 4 per low half (no crossing: `W_RESP` at `E_1`, B at `E_2`, next accept
at `E_4`), plus `intercycle = 8` posedges after every `w`/`r` command (bench
274, 320): about 22 cycles per 64-bit register write, and a launch descriptor
is ten of them (`_put_request`, `u50.py:521-531`: SEQ, OP, A0-A5, DEADLINE,
DOORBELL). Reads cost the same order (`R_WAIT` two device cycles, 392, 436-439).
The absolute cycle at which a batch starts is set by the bench's poll of the
command file every `pollcycles = 20` posedges (bench 345) and by the host's
wall clock; see sec 4.1.

**Doorbell to dispatcher.** With `reg_we` for 0x48 during `d` (`reg_wdata[0]`,
`state == IDLE`, `host_state == IDLE`, `all_idle`, `req_seq > served`, mmio
130; op 11 and `flags_ok`, 162-171): at `E_d` `cmd_valid <= 1`, `state <= ISSUE`,
the `cmd_*` fields, `launch_start <= device_clock`. During `d+1`: `cmd_valid`
-> (`&& memory_drained`) shell `cmd_valid` -> `grid_valid = cmd_valid && !bad_pc
&& !code_busy && !bad_host_pending && !invalid_done` (shell 188) -> dispatcher
`launch_valid = grid_valid && !host_busy && !maintenance_requested &&
!maintenance_busy` (`tnca_top.v:321`), `launch_ready = !running && !done_valid`:
**`L = d+1`**, and the first core launches at `d+5` (sec 5.1). At `E_{d+1}` the
MMIO sees `cmd_ready` and drops `cmd_valid`, `state <= WAIT` (183). So from the
edge that accepts the doorbell's high half: `reg_we` at cycle 5, `L = 6`, first
`launch_valid` at 10. `launch_end - launch_start = (i+3) - d = done_cycles + 2`
(sec 5.3). All of this is identical for DISPATCH_STAGES = 0 up to `L`; only
the `+4` after `L` is the board's.

**Code store writes** (host window at `host_addr < CODE_BYTES`, shell 81, 97):
`code_fire` during `f` (83) is pushed into every core's `commands` link at
`E_f` (113-116); `command_valid` during `f+2`; the RAM is written at `E_{f+2}`
(132-136); `reply_valid` during `f+3` (119-121); `replies` link: `code_reply_valid`
during `f+5` (125-127); the collector (144-159) sets `seen` at `E_{f+5}`, so
`code_pending = host_rsp_valid` during `f+6` and `active` (= `code_busy`) is
high during `f+1 .. f+6`. STAGES = 0: `host_rsp_valid` during `f+1`. `code_busy`
blocks `cmd_ready` (91), `grid_valid` (188), `host_ready` (97) and `all_idle`
(71), so a launch cannot be accepted while a code word is in flight; it does
not touch the fetch path. The MMIO's host window (123-129, 178-182) adds one
cycle each way around this (`host_state` ISSUE -> WAIT -> IDLE).

**Data words through the host window** (`host_addr >= MEM_BASE`) take the
same coherent path as on the sim bench (UNCORE_TIMING sec 7: tile 0's core NI,
the home slice, and on a miss the home link, port, outlet link, adapter and
line model of sec 2-4), one word per round trip, so a host upload or download
costs the full board miss latency per missed line (sec 4.4) plus the register
traffic per word. The bench also offers a DMA path (`b/d/B/D` commands, bench
258-269, `tb/fpga/tnca_multicore_dma.vh`) that writes the line models directly
and is only legal when `MAINTENANCE` and `dut.device.all_idle`;
`MulticoreMMIOLink.hbm_read/hbm_write` use it (multicore_mmio_sim.py:50-77),
but `MulticoreMMIOBackend` takes that route only with `hbm_dma = True`
(multicore_mmio.py:19, 66), which `record_rtl.py` does not pass (138): the
recordings move data through the host window and never flush (op 18) between
launches, as on the sim bench.

## 7. What the simulation-bench model must change for the board configuration

Checklist against `tools/performance_harness/systemc/src/cycle_uncore.inc`
(the sim-bench uncore: PORTS = 1, `tnca_dram_model`, no links). Each item names
the RTL parameter that turns it on; "unchanged" items are listed so that nobody
adds them.

- [ ] **Slice: unchanged.** SETS = 4, WAYS = 2 (bench 42), and NUCA_REGISTER_VICTIM_SELECT = NUCA_CACHE_DATA_PIPELINE = NUCA_REGISTER_VICTIM_RANK = 0, DCO_* = 0 (xo defaults, sec 0): the sim-bench engine (3-cycle miss allocation, no E_ALLOC/E_RANK, combinational TMU) applies as is. Only an alignment against the U280 image itself would need the "board" slice column.
- [ ] **Mesh links, NOC_LINK_STAGES = 2** (`tl_noc_link_stages`, already read by the model, cycle_uncore.inc:61-64): per hop head-to-head 3 cycles; 2 x 2 flits of link buffering per hop per VN, `in_ready` per stage from its own count; sec 1.1-1.3.
- [ ] **Home link, MEM_HOME_STAGES = 2:** request side two 2-deep FIFOs between the slice's E_MREQ/E_WBSND/E_MSEND and the port; the slice's `mem_req_ready` is stage 0's `count < 2`, so the engine leaves E_MREQ as long as fewer than 2 requests wait in stage 0 (4 in the link), regardless of the port. Response side +2 cycles, never stalled. Sec 1.2, UNCORE_TIMING sec 4.1.
- [ ] **No memory arbiter among homes, MEM_PORTS = 3 with `memory_active_ports = 3`:** home `h` -> port `h`, three independent chains, zero-cycle `tnca_mem_ports` and `tnca_memory_mode`; replace the single arbiter (`arb_rr/arb_held`, cycle_uncore.inc:47) by three ports keyed by home. Keep the arbiter code for `single_port = 1` (`TNCA_RTL_MEM_PORTS = 1`: all homes on port 0 with round robin + hold, sec 2.3), selected per record.
- [ ] **Outlet link, LINK_STAGES = 2:** another 2 x 2-deep request FIFO chain and +2 response registers per port between the mode block and the adapter; back-pressure from the adapter is absorbed by 4 requests here and 4 in the home link before the slice stalls. Sec 1.2.
- [ ] **Adapter per port, `tnca_line_axi` (AXI_OUT_N = 8, DYNAMIC_LIMIT = MEMORY_CONTROLS = 1, `admission_limit = memory_outstanding`, 8 unless `TNCA_RTL_OUTSTANDING`):** 8 slots, lowest free index; `req_ready` = no issue in progress, a free slot, no same-line hazard (any active write on the line, or any active access if this is a write), `active_count < admission_limit`; issue occupancy 2 cycles per read / 3 per write at best, extended by every cycle an AXI ready is low; `rsp_valid` one cycle after the last R beat; a write slot stays active until B and produces no response; head-of-line blocking of the port's FIFO behind a refused request. Sec 3.
- [ ] **AXI slave = `tnca_axi_line_model` per port (bench 149):** LATENCY 25 / 44 / 63 by port; read skew `(slot*11) % 31`, write skew `(slot*7) % 23` plus WRITE_DELAY 15; registered random readies `awready` 64 %, `wready` 59 %, `arready` 69 % (`awready` 0 for the model's first 12 cycles); one R beat per cycle, random pick among due reads with the ascending-scan bias of sec 4.2, beats of different IDs interleave; B for the lowest due write index; memory updated at the B handshake; `$random(seed = 123)` per port with 3 + max(0, n_due - 1) draws per posedge. To predict rather than replay, the model must advance the same generator from the same origin (model `cycle` starts 2 posedges before the device's `rst_n`, sec 0 and 8.1) and reproduce every AR/AW/W handshake cycle exactly, as with the sim bench's LFSR (UNCORE_TIMING sec 5). Sec 4.
- [ ] **Replay records on this bench** (`tnca_core_trace.v:21-23, 134-144`, `-DTNCA_TRACE_BOARD`): `X` = an adapter's `req_valid && req_ready` (per port, with `we/addr/src/tag`), `Y` = an adapter's `rsp_valid` (per port). They are at the adapter, i.e. 4 cycles after the slice's request leaves and 4 cycles before the slice's `mem_rsp_valid`; a replaying model places `X` at the adapter and delivers `Y + 4` to the slice. There is no record of B, so a replay cannot know when a writeback's slot frees: either model the B (sec 4.3) or treat hazards as absent and note it.
- [ ] **Dispatch, DISPATCH_STAGES = 2:** first core `launch_valid` at `L+4` (then `L+5`, `L+6`); re-launch of a core that went idle at `i` at `i+6`; grid `done_valid` at `i+3`; `done_cycles = i+2-L`. Model's `dispatch_cycles` / `launch_cycles` must be checked against these, not the sim bench's `L+2` / `i+2` / `i+1`. Sec 5.
- [ ] **Host-visible completion gating (`memory_drained`):** the MMIO's completion, `launch_end` and the acceptance of the next launch wait for every adapter slot (including writebacks waiting for B) and every outlet link to empty; `done_cycles` does not. If the harness compares `done_cycles`, nothing to model; if it compares `launch_end - launch_start` (= `done_cycles + 2` + drain) or the inter-launch gap, model the drain (sec 5.3).
- [ ] **Launch arrival:** `L = d+1` for the doorbell's `reg_we` cycle `d`; `reg_we` is 5 cycles after the AXI-Lite bridge accepts the high half; the absolute cycle is otherwise host-wall-clock dependent (sec 6, 4.1). Only the relative offsets above are reproducible without the journal.
- [ ] **Code uploads, CODE_LINK_STAGES = 2:** 7 cycles of `code_busy` per code word (vs 2) before the launch; fetch unchanged. Only matters if the model accounts for the time before `L`. Sec 6.
- [ ] **Host data words, unchanged in kind:** same coherent path and same slice state left behind as on the sim bench; each missed line now costs the sec 4.4 latency; no flush between launches in the recordings (sec 6). The DMA path and op 18 are unused unless `hbm_dma` is set.
- [ ] **MAINTENANCE = 1, unchanged for launches:** `maintenance_requested/busy` only gate `grid_ready`, `host_allow` and `cmd_ready` while a flush runs (UNCORE_TIMING sec 3.8), which never overlaps a launch.

## 8. Points not settled by reading

1. The origin offset between the line model's `cycle` and the trace's
   `cycle` (`tnca_core_trace.v:33, 163`): I read the model's first counted
   posedge as the one right after `rst_n_async` rises and the trace's cycle 0
   as the first posedge with the TILE's `rst_n` high (two posedges later), so
   model `cycle = trace cycle + 3` inside the model's `always` block; and
   `awready` can first be 1 during trace cycle 11. One waveform settles it.
2. Whether the recordings' `memory_active_ports` / `memory_outstanding` are
   really 3 / 8: `start()` writes what `TNCA_RTL_MEM_PORTS` /
   `TNCA_RTL_OUTSTANDING` say, else the capability (3, 8); `record_rtl.py`
   does not set them, `run_graphs.py:45-46` (UNCORE_TIMING sec 4.3) used 1 / 1
   for a different flow. Confirm per record (the MMIO 0x110 write is in the
   journal if one was kept).
3. Short-circuit evaluation of `&&`/`||` inside the model's read-pick loop
   (74) under Icarus `-g2012`: it decides whether a non-due slot consumes a
   `$random` draw. The count "3 + max(0, n_due - 1)" assumes it does not.
   Check with a two-read test and the resulting ready pattern, or make the
   model consume the stream by replaying `X`/`Y` instead.
4. `tnca_core_trace.v`'s `rst_n` source for `cycle` (only `clk` was read as
   `TILE(0).u_core.clk`, line 40); affects item 1 by exactly the reset
   synchroniser's 2 cycles if it is the asynchronous reset instead.
5. The image-vs-bench parameter difference of sec 0 (seven NUCA_*/DCO_*
   parameters at 0 on the bench, 1/1/1/1/1/1/2 on the image) is read from
   `image.tcl:1` and `tnca_multicore_hbm_xo.v:8-14`; whether anything in the
   flow (`flow/fpga/*`) injects them into the bench was not checked, and
   `multicore_mmio_sim.py:86-89` passes only the ten `-P` values listed in
   the header.
6. The AXI-Lite cycle counts of sec 6 (10 / 4 cycles per bus write) depend
   on the bench task's negedge/posedge sequencing (bus_write, 207-220) and
   were derived, not simulated; they do not affect any offset after `L`.

## 9. R41 (a076cd64)

### 9.1 The bench is the image

`record_rtl.py --device` builds the board-top bench with
`tnca_image_params.vh` (`TNCA_IMAGE_PARAMS`), so the dut takes the image's
own parameter list: 32 sets x 8 ways and every NUCA / DCO value. Sec 0's
bench-versus-image difference no longer exists; what aligns against this
bench is the image's uncore. The runtime sets the memory mode to every outlet
with OUT_N lines admitted (`multicore_mmio.py`: `physical_ports`,
`maximum_outstanding`) unless TNCA_RTL_MEM_PORTS / TNCA_RTL_OUTSTANDING say
otherwise.

### 9.2 The adapter: OUT_N 32, one beat per line, pipelined issue

`tnca_line_axi.v` with AXI_OUT_N 32 (5-bit IDs), AXI_DATA_W 512 (BEATS 1) and
PIPELINED_ISSUE 1:

- `req_ready = !pending_valid && free slot && !hazard && legal && admission`
  (124): the issue register no longer refuses a request; a one-entry pending
  slot takes a request accepted while the issue register is busy and not
  finishing.
- A read is AR plus one R beat; a write is AW plus one W with `wlast`; with
  every AXI ready high an issue finishes in the cycle it is valid, so the
  pending slot is never occupied and the adapter accepts one request per
  cycle (R40: a read every 2, a write every 3 cycles).
- The read's response is the cycle after its single R beat; B is still
  refused while a final R beat arrives (145); a write's slot is held until
  its B.
- HBM_STRIPED changes only the AXI address (a bijection of the 8 GiB window);
  the hazard check compares the logical line address (35 bits).
- The skews of sec 4 extend to 32 IDs: reads `(id*11) % 31`, writes
  `(id*7) % 23`; the line model's latencies stay 25 / 44 / 63.

The model's adapter keeps sec 3's structure with `uncore_axi_slots`,
`uncore_axi_data_w` and `uncore_axi_pipelined_issue`; under replay X / Y are
the recorded accept and response cycles, as before.

### 9.3 Dispatch admission (R41 F06)

`tnca_grid_dispatch.v` validates a launch in four registered stages after the
accepting edge before the first enqueue: the first enqueue moves from L+1 to
L+5 and the first core launch from L+4 to L+8; re-launch (i+6) and done (i+3)
are unchanged (model: `uncore_dispatch_admission=4`).

### 9.4 Not reproduced

The bench's line model now reloads its seed at every launch (the edge ending
L+1, from a separate `always` block racing the model's own draws); prediction
does not reproduce the line model's random readies at all, and replay takes
the recorded accept and response cycles, so neither depends on the order.
