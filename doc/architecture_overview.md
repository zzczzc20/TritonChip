# TNCA Architecture Overview

Status as of 2026-10-08 (branch `r38`). This document describes the TNCA hardware as it is built and measured today, how it relates to the original design, and what comes next. Its companion, [`compiler_overview.md`](compiler_overview.md), covers the compiler, the runtime and the PyTorch path.

> **Supersedes the September 2026 design draft.** The previous `architecture_overview.md` was a design study for a 16-tile TNCA-v1 (in Chinese). It is preserved unchanged at [`design/tnca_v1_architecture_draft.md`](design/tnca_v1_architecture_draft.md). Section citations such as "`architecture_overview.md` §11.3" or "the twelve frozen decisions" in code comments and in documents written before 2026-10-08 refer to that draft, whose section numbers are unchanged.

---

## Contents

1. [Summary](#1-summary)
2. [Design intent and what is built](#2-design-intent-and-what-is-built)
3. [System organisation](#3-system-organisation)
4. [Core microarchitecture](#4-core-microarchitecture)
5. [Memory and address map](#5-memory-and-address-map)
6. [Instruction set](#6-instruction-set)
7. [Profiles](#7-profiles)
8. [Observability: counters, profiler and cost model](#8-observability-counters-profiler-and-cost-model)
9. [Device interface and platforms](#9-device-interface-and-platforms)
10. [Measured behaviour](#10-measured-behaviour)
11. [Verification and evidence](#11-verification-and-evidence)
12. [Known limitations](#12-known-limitations)
13. [Next steps](#13-next-steps)
14. [Source map](#14-source-map)

---

## 1. Summary

TNCA (Tensor-Native Compute Array) is an accelerator architecture designed to run Triton kernels without a GPU-style shared memory. The design intent is that registers are the compute working set, a single shared cache level (NUCA) holds the data, a tensor transfer engine moves tiles described by TensorMap descriptors, and a few role contexts per core drive an asynchronous producer/consumer pipeline.

What exists today is a verified single-core implementation of that core, plus the tools to compile real PyTorch programs onto it and measure them:

| | State |
|---|---|
| Core RTL | One in-order core: 16-lane × 32-bit vector unit, 16×16 FP32-accumulating matrix unit, tensor transfer engine, mbarriers, atomics, SFU, 2 role contexts. Verilog-2005, simulated with Icarus Verilog. |
| Multi-tile RTL | A 2×2 mesh with NUCA slices and a NoC (`rtl/tnca_top.v`). It is exercised by RTL tests only; the runtime does not use it. |
| ISA | TNCA-ISA v0.1: fixed 64-bit instructions, 20 major groups, 169 instructions, one machine-readable source. |
| FPGA | The single core (`core-vrf-sync-v1`) runs on an Alveo U50 and, since 2026-10-08, an Alveo U280. The kernel clock is 150 MHz. The loaded images divide it by 4, for a 37.5 MHz device clock. Builds from the R38 timing work divide by 3, for 50 MHz ([`r38_board_bringup.md`](r38_board_bringup.md) §6). 512 KiB of on-chip memory. |
| Software | Triton → TNCA compiler; PyTorch `torch.compile` → Inductor → Triton → TNCA, with no host fallback on any tiny- or small-tier ArchBench point. |
| Measured | ArchBench tiny (27 points) and small (25 points) pass on the U280. The board's cycle counts equal the RTL simulator's, launch by launch, on every tiny-tier point. The GB10 GPU passes all 76 points under the same protocol. |

TNCA is correctness-first today. Strict-bound efficiency η is about 0.001–0.2 per point. Section 10 shows where the time goes; most of it is register hazards and a single 8-byte memory port, not arithmetic.

## 2. Design intent and what is built

The design draft ([`design/tnca_v1_architecture_draft.md`](design/tnca_v1_architecture_draft.md)) fixed six principles and a reference configuration. The table below says, for each one, what the RTL implements now.

| Design element (draft §) | Draft reference | Implemented now |
|---|---|---|
| Storage hierarchy RF ↔ NUCA ↔ DRAM, no SPM (§1.1) | No addressable scratchpad | **Holds.** There is no scratchpad. Single-core profiles have no cache; the core talks to one flat memory. The 2×2 system has NUCA slices and a DRAM model. |
| Tiles and mesh (§3.1) | 16 tiles, 4×4 mesh | 1 core (all runtime profiles). A 2×2 mesh exists in RTL tests (`system-2x2`). 4×4 is explicitly unsupported (audit F05). |
| Role contexts (§1.3, §4) | ≤ 4 vector control contexts per core | `NUM_RCTX` parameter: 2 in `core-sim`, 4 in `core-4r`. The runtime launches with role mask 1, so one context per program. |
| SIMD width (§3.1) | 16 lanes | **16 lanes × 32 bits.** |
| Matrix unit (§6) | M16×N16, K16/32/64, 256 MAC/cycle per tile | M16×N16, K = 16/32/64, **16 MAC/cycle** (1/16 of the reference). FP32 accumulation, one rounding per step. |
| Vector RF (§5) | 32 KiB per tile | **8 KiB:** 128 physical registers × 16 lanes × 32 bits. Optional 4-bank synchronous organisation with an operand collector (`core-vrf-sync-v1`). |
| Accumulator RF (§5.1) | 16 KiB per tile | **16 KiB:** 16 slots × 16 × 16 × FP32. |
| TensorMap (§7) | 256-byte descriptor; semantics separated from hints | 256-byte descriptor; **rank 1–5** with signed coordinates and zero/NaN/fault padding — the TMA contract Triton exposes on NVIDIA GPUs, with no alignment requirement on base or strides; no hint fields act yet. |
| NUCA cache (§8) | 512 KiB/slice, 8 MiB, 96 MSHRs; TensorMap-aware replacement and bypass | 2×2 system only: 32 KiB/slice (64 sets × 8 ways × 64 B), 8 MSHRs. DCO ([`dco.md`](dco.md)): a Tensor Management Unit with compiler-registered nAcc, dead-block prediction, anti-thrashing and LRU replacement, dynamic, tensor and gqa bypass. Each is selected per run; the default stays round-robin. No admission control or MSHR-aware arbitration. |
| Transaction control (§9) | Credits, MSHR-aware arbitration, admission | Not implemented. |
| Async protocol (§13) | mbarrier with ready/free events | **Implemented:** 32 mbarriers per core, counting arrivals and operations separately; the matrix unit signals result-ready and source-free separately. |
| Clock (§3.1) | 1 GHz modelling assumption | Not a physical design: 37.5 MHz on the loaded FPGA image, 50 MHz for new builds. Synthesis numbers are structural proxies, not PPA (roadmap §1.2). |

The draft's own staging (§21) placed "prove function on one core" first and "Tensor DMA, RF windows, matrix" second. Both are done on a single core. Stage 3, a 2×2 mesh with reliable NUCA, exists in RTL but not in the runtime. Stages 4–5 (the cache-policy ideas and the 4×4 scale-up) have not started.

## 3. System organisation

### 3.1 The device as the runtime sees it

The public hardware top is `rtl/tnca_shell.v`: one core plus the ports a host needs.

```text
             host (runtime over a transport: file mailbox, socket, or PCIe registers)
                │ command: NOP / LAUNCH / QUIESCE        ▲ completion: OK / FAULT / TIMEOUT / BADCMD
                ▼                                         │
 ┌─────────────────────────────── tnca_shell ─────────────────────────────────┐
 │  command decode, device deadline, perf-counter mux (PERF_READ)            │
 │  ┌────────────────────────── tnca_core ──────────────────────────────┐    │
 │  │ F: RCTX select ─ D: decode ─ R: scoreboard ─ X: read/execute/WB    │    │
 │  │  scalar ALU + mul64 │ vector ALU │ vector FPU │ SFU │ matrix + ACC  │    │
 │  │  scalar / predicate / vector RF        mbarriers   descriptors      │    │
 │  │  LSU (8 beats in flight) ─┐   ┌─ tensor transfer engine (txf)       │    │
 │  │                           └─ mem_mux ─┘  (txf has priority)         │    │
 │  └──────────────────────────────┬──────────────────────────────────────┘    │
 │        instruction fetch port   │   64-bit data port (1 request/cycle)       │
 └─────────────────────────────────┼────────────────────────────────────────────┘
                                   ▼
            unified memory: 65,536 × 64 bit = 512 KiB (simulator model or FPGA BRAM)
```

A launch names a PC, an argument block, a role mask and the program id/count. A grid is launched program by program (`serial-grid`), so device time for a grid is the sum of its programs' cycles and not a concurrent makespan.

### 3.2 The multi-tile system (RTL only)

`rtl/tnca_top.v` instantiates an NX×NY mesh (default 2×2). Each tile (`rtl/tile/tnca_tile.v`) has a core, a NUCA slice (`rtl/nuca/tnca_nuca_slice.v`) and network interfaces. The NoC is in `rtl/noc/` (mesh, routers, core and slice NIs), with a DRAM model behind a memory arbiter. One Tensor Management Unit (`rtl/nuca/tnca_tmu.v`) serves all slices: it holds the tensor metadata and dead-tile FIFO that DCO's replacement and bypass policies read, and software reaches it through an uncached register window at `0x7FFF0000` ([`dco.md`](dco.md)). The `system-2x2` profile describes it, with the SFU disabled. CI builds and runs it (`make top`). It is the starting point for the multi-core work in §13. Neither the runtime nor ArchBench targets it.

## 4. Core microarchitecture

### 4.1 Pipeline and issue

`rtl/core/tnca_core.v` is the only core module. All profiles are parameterisations of it.

- **Stages:** F (round-robin selection among role contexts; synchronous fetch with a one-entry fetch buffer), D (decode), R (scoreboard), X (register read, execute, writeback through a registered latch).
- **Issue:** single issue (`issue.width = 1` in every target file). The FU class is kept in the decoder for a future dual-issue option.
- **Ordering:** in order within a role context. There is no reorder buffer or renaming. Writeback slots are booked in an 8-deep reservation shift register.
- **Control flow:** predicted not-taken. A redirect squashes only the younger entries of that role context.
- **Role contexts:** a context blocked in `mb.wait` stops fetching, and the others continue.

### 4.2 Architectural state

| State | Size | Notes |
|---|---|---|
| Scalar RF | 64 × 64 bit per RCTX | `S0` reads as zero; 3 read / 1 write ports |
| Predicate RF | 16 × 16 bit per RCTX | `P0` is all-ones; one bit per lane |
| Vector RF | 128 physical × 16 lanes × 32 bit = 8 KiB per core | 64 logical registers per RCTX through a programmable base; encoding room for 256 |
| Accumulator RF | 16 × (16 × 16 × FP32) = 16 KiB | Read and written by the vector unit and the matrix unit |
| Tensor descriptors | 16 | Snapshots of 256-byte TensorMaps |
| RF windows | 32 | Destinations of tensor transfers |
| mbarriers | 32 per core | Shared by all role contexts |

### 4.3 Data types

- **Vector:** 32-bit integer and binary32 floating point. Conversions among FP32, BF16, FP16, E5M2 and E4M3, and between int32 and FP32 (ISA §8.4).
- **Matrix:** FP16, BF16, E5M2 and E4M3 inputs with FP32 accumulation.
- **Scalar:** 64-bit integer only. There is no scalar floating point.
- **Not in hardware:** integer or floating-point divide (the compiler uses SFU reciprocal plus Newton steps), integer MMA (INT8 goes through FP16 MMA, exact for K ≤ 1024), and `v.pack`/`v.unpack` (reserved encodings). See [`r37_isa_gaps.md`](r37_isa_gaps.md).

### 4.4 Functional units

| Unit | Behaviour in the RTL |
|---|---|
| Scalar ALU | 1 cycle; 64-bit multiplier pipelined over 3 cycles |
| Vector ALU | 16 lanes, 1 cycle |
| Vector FPU | 16 lanes, 2-cycle latency; lane reductions take log2(16) = 4 steps on the FMA array |
| SFU | One shared slice computing exp2, log2, reciprocal and rsqrt; one lane per cycle, 6-cycle latency, ≤ 2⁻²² relative error, subnormals flushed. A 16-lane op occupies it for VLEN + 8 cycles. Generated from XLS (`rtl/gen/`). |
| Matrix | `C16×16 += A·B`, K = 16/32/64, 16 MACs per cycle (one k step per 16 cycles); separate result-ready and source-free completion events |
| Tensor transfer engine | `t.ld.async` / `t.st.async` of a rank-1–5 box between memory and an RF window, asynchronously, one completion per command on an mbarrier. Coordinates are five consecutive scalar registers, signed and of any alignment; base and strides need no alignment either. A misaligned row is assembled from two memory words per 8-byte unit. Out-of-bounds elements are filled on a load (zero, or the dtype's NaN) and not written by a store (byte enables). The store reads the window through its own VRF port. Box rows must be whole 8-byte units, block dimensions 1–256. `core-xpose-v1` adds a transposed window write for 16-bit elements (§7). |
| LSU | Ordinary loads, stores, gathers and scatters, coalesced onto 64-bit words; up to 8 beats outstanding, completed out of order by tag; one instruction in flight |
| Memory mux | Merges the LSU and the transfer engine onto the core's single 64-bit data port; the transfer engine wins |
| Banked VRF (option) | 4 banks, each 1R1W with synchronous read; an operand collector with 4 entries × 3 sources (`core-vrf-sync-v1`, the board profile) |

### 4.5 Synchronisation, atomics and faults

- `fence` drains the LSU. It does not wait for asynchronous transfers; those complete through mbarriers.
- mbarrier operations: `init`, `arm`, `arrive`, `wait`, `test`, `inval`. Arrivals and outstanding operations are counted separately (draft §13 semantics).
- Atomics are 4 or 8 bytes, naturally aligned, at device scope only. System scope raises `ILLEGAL_INSTRUCTION`. The atomic functions are shared between the FPGA memory and the NUCA slice (`rtl/common/tnca_amo.vh`).
- Faults stop the program with a cause and a PC (`rtl/common/tnca_defs.vh`): illegal instruction (0x01), misaligned PC (0x02), access fault (0x03), descriptor fault (0x04), barrier fault (0x05), resource fault (0x06) and user trap `TRAP_USER` (0x40). The device deadline is reported as a TIMEOUT completion. The compiler uses user traps for `tl.device_assert` (0xA55E7) and for failed 64-bit→32-bit index narrowing (0xA5564). A fault leaves the session usable (conformance case CONF-FAULT).

## 5. Memory and address map

| Property | Value |
|---|---|
| Size | 65,536 words × 64 bit = 512 KiB (`ADDR_W` = 16), shared by instruction fetch and data |
| Timing | Fetch 1 cycle. Data: fixed 2-cycle latency, one request per cycle, byte enables, read-first. The simulator's default model uses latency 1; the `/slow` variants use latency 7 with random stalls and reordering, to shake out protocol bugs. |
| Bandwidth | 8 bytes per cycle per core: one 64-bit port shared by the LSU and the transfer engine |
| Cache | None in the single-core profiles. NUCA slices only in `system-2x2`. |
| FPGA | Block RAM (`rtl/fpga/tnca_fpga_mem.v`), uninitialised at configuration; the runtime uploads code and data |

The runtime's map ([`compiler_overview.md`](compiler_overview.md) §8):

```text
0x00000 ┌────────────────────────────┐
        │ code (modules, 64-B align) │   128 KiB
0x20000 ├────────────────────────────┤
        │ heap: argument storages,   │   384 KiB
        │ argument blocks (256-B),   │
        │ TensorMaps, scratch        │
0x80000 └────────────────────────────┘
```

The 384 KiB heap bounds what one launch can touch. The small ArchBench tier was sized to it (largest launch 280 KiB). The large tier (Llama-3.2-1B shapes) does not fit and is recorded as `capacity-exceeded`.

## 6. Instruction set

Reference: [`isa/TNCA-ISA-v0.1.md`](isa/TNCA-ISA-v0.1.md), machine-readable at `isa/tnca-isa-v0.1.json`.

- **Encoding:** fixed 64-bit little-endian words, 8-byte aligned. `major[63:56] func[55:48] d[47:40] a[39:32] b[31:24] c[23:16] pred[15:12] X[11] PN[10] mod[9:0]`. `X` reserves an extension word; v0.1 uses none. Reserved bits must be zero.
- **Groups (20 majors):** SYS, SALU, CTRL, VINT, VFP, CVT, PRED, SHUF, RED, SFU, LSU, ATOM, TDESC, TXFER, THINT, MATRIX, ACC, MBAR, ROLE, DEBUG. THINT, ROLE and DEBUG are reserved and contain no instructions yet.
- **169 instructions:**

  | Group | Count |
  |---|---|
  | SALU | 36 |
  | VINT | 34 |
  | ATOM | 13 |
  | PRED, RED | 11 each |
  | VFP | 10 |
  | CTRL | 9 |
  | SHUF | 7 |
  | LSU, MBAR | 6 each |
  | CVT, ACC | 5 each |
  | TDESC, SFU | 4 each |
  | SYS, TXFER | 3 each |
  | MATRIX | 2 |

  The compiler emits 111 of them ([`support_matrix.md`](support_matrix.md)).
- **One source, many consumers:** the authoring tables are `tools/asm/tnca_isa.py`; everything else is derived from them, and `make compiler` fails if any copy is stale.
  - `tools/isa/export_schema.py` writes the JSON.
  - `tools/isa/gen.py` writes the LLVM-style opcode tables and [`isa/opcode-map.md`](isa/opcode-map.md).
  - `tools/isa/gen_target.py` writes the per-profile target files `compiler/targets/*.json` from the schema and the RTL parameters.
  - `tools/isa/gen_support_matrix.py` writes the support matrix.
  - `tools/isa/check_consistency.py` checks the hand-written RTL decode defines (`rtl/common/tnca_defs.vh`) against the tables.
- **Known documentation drift:**
  - The ISA text marks major 0x21 (atomics) as reserved, and the A/W/D/B register classes as Stage 2; both are implemented.
  - It describes the vector-register mapping as the identity, but the RTL uses a programmable per-context base.

  These are recorded here and need an ISA v0.2 text revision (§13).

## 7. Profiles

A profile is a named RTL configuration plus its capability record. Profiles are defined in `spec/r1_profiles.json`; the target files are generated from them.

| Profile | What differs | Used by |
|---|---|---|
| `core-sim` | Reference: 1 core, 2 RCTX, 64 vector registers per RCTX, behavioural VRF, SFU | Default simulator profile, R3.7 TileBench, tiny ArchBench on RTL |
| `core-vrf-sync-v1` | Same ISA; synchronous 4-bank VRF with operand collector (R3 v2) | **The FPGA board.** ArchBench board runs; the simulator bench of the same profile is the board's cycle reference. |
| `core-xpose-v1` | Adds `tensor_transpose`: the transfer engine writes a transposed window (16-bit elements) | R38 three-layer case S1-07: GEMM with a transposed operand runs 1.46× faster for a +11.2 % structural gate proxy |
| `core-4r` | 4 RCTX, 32 vector registers each | RTL tests of four-role elaboration |
| `system-2x2` | 4 tiles, mesh, 4 NUCA slices, DRAM model, no SFU | RTL tests only |

Each runtime profile also has a `/slow` simulator variant (memory latency 7, random stalls and reordering) for robustness testing. The compiler reads the capability record (for example `tensor_transpose`, SFU present, atomics) and refuses to emit what a profile lacks, rather than emitting it and failing at run time.

## 8. Observability: counters, profiler and cost model

### 8.1 Hardware counters

Twenty 64-bit counters (`spec/r38/counters.json`). They are cleared only by reset and wrap. The selection logic is generated by `tools/r38/gen_counters.py` into `rtl/common/tnca_perf_sel.vh` and `tnca_perf_count.vh`.

| Index | Counter | Meaning |
|---|---|---|
| 0 | `perf_cycles` | Cycles with any role context running |
| 1 | `perf_issued` | Instructions issued |
| 2 | `perf_stall_total` | Cycles in which nothing issued |
| 3–13 | `perf_x_hazard`, `_lsu`, `_fpu`, `_mtx`, `_txf`, `_fence`, `_arrive`, `_rsv`, `_wb`, `_vrf`, `_vrfwb` | Exclusive stall attribution by fixed priority; these sum to `perf_stall_total` |
| 14–16 | `perf_stall_hazard`, `_lsu`, `_wb` | Overlapping view of the same stalls |
| 17 | `perf_squash` | Redirects |
| 18–19 | `perf_mem_rd_beats`, `perf_mem_wr_beats` | 8-byte beats on the data port |

Reading a counter:
- **Simulator and board:** transport operation 7, `PERF_READ`, with the counter index in, low/high 32 bits out. It is served only while the device is idle, and a device without the operation answers `REFUSED`.
- **Runtime:** with `counters=True` (or `TNCA_COUNTERS=1`), the runtime brackets each program with counter reads. The reads happen outside the launch's measured window, so they add host time but no device cycles; R38 profiler test 5 checks this.

### 8.2 Profiler and cost model

- **Profiler:** `tools/r38/profiler_tests.py` checks that the exclusive stall counters sum to the total, that counters only increase, and that enabling them changes no cycle count. It runs on any backend. It passed on `core-sim` and `core-vrf-sync-v1` in simulation, and on the U280 (full run, 2026-10-08).
- **Cost model:** `tools/r38/costmodel/` is a fitted cycle predictor (`programs × (a + b·trips + c·tail)`), validated on held-out kernels in Stage 1 (S1-06). It is not a microarchitectural simulator.
- **Strict peaks:** ArchBench bounds use peak rates computed from the target file and RTL constants (`tools/r38/archbench/peaks.py`). They are identical for every single-core profile:

  | Rate | Peak |
  |---|---|
  | FP lane operations | 32/cycle (vector 16 + matrix 16) |
  | MMA MACs | 16/cycle |
  | Integer lane operations | 16/cycle |
  | SFU lanes | 1/cycle |
  | Memory | 8 B/cycle |
  | Issue | 1/cycle |

## 9. Device interface and platforms

### 9.1 Transport protocol

All backends speak one request/response protocol (`python/tnca/transport.py`, `spec/r2_transport_v1.json`):

| Op | Name | Purpose |
|---|---|---|
| 1 | UPLOAD | Host → device memory |
| 2 | DOWNLOAD | Device → host memory |
| 3 | LAUNCH | PC, argument block, role mask, program id/count (16 bits per axis) |
| 4 | QUIESCE | Wait for the device to be idle |
| 5 | SHUTDOWN | End the device process (a no-op on the board) |
| 6 | PING | Liveness |
| 7 | PERF_READ | Read one hardware counter (added in R38; the JSON spec still says v1.0 and lists ops 1–6, see §13) |

Status codes: OK 0, FAULT 1, TIMEOUT 2, REFUSED 3. Sequence numbers increase monotonically. Late responses, and responses from an earlier epoch, are discarded.

### 9.2 Backends

| Backend | Device | Evidence level of its measurements |
|---|---|---|
| `sim-file` | RTL simulator (Icarus, about 500 cycles/s) behind a file mailbox | `measured-rtl` |
| `sim-socket` | The same RTL behind a TCP relay | `measured-rtl` |
| `fpga-u50-sim` | The board's register protocol and FPGA top, simulated on RTL | `measured-rtl` |
| `fpga-u50` | Alveo U50 over XRT (PCIe register window) | `measured-board` |
| `fpga-u280` | Alveo U280; same kernel and backend code as the U50 | `measured-board` |
| `fpga-prototype` | Interface only; opening it is an error | — |

No backend claims `platform-verified`; the registry reserves that state. A backend that needs hardware the machine lacks refuses to open. It never substitutes the simulator.

### 9.3 FPGA implementation

- **Tops:**
  - `rtl/fpga/tnca_fpga_top.v`: shell + MMIO mailbox + BRAM memory + reset synchroniser. Defaults to the banked VRF, i.e. `core-vrf-sync-v1`.
  - `rtl/fpga/tnca_axi_lite.v`: a 32-bit AXI4-Lite slave, crossing from the shell clock to the device clock.
  - `rtl/board/tnca_u50_xo.v`: the Vitis kernel top. It is the only file with a Xilinx primitive: a BUFGCE_DIV ÷4, plus a 64-cycle power-on reset.
- **Mailbox registers** (`rtl/fpga/tnca_mmio.v`):

  | Offset | Contents |
  |---|---|
  | 0x00–0x48 | Request: sequence number, op, arguments, doorbell |
  | 0x50–0x68 | Response |
  | 0x70–0x90 | One-word payload window: address, data, byte enable, control, status |
  | 0x98 | Magic `0x0001_0000_544E_4341` |
  | 0xA0 | Served-request count |

  A doorbell is acted on only when its sequence number is greater than the served count.
- **Clocking:** the kernel clock is requested at 150 MHz and divided by 4 inside the kernel, giving 37.5 MHz. The first U50 link at the platform default of 300 MHz failed on 173 setup endpoints, 131 of them outside the design's control. The device domain had already closed with +6.494 ns at 37.5 MHz. The measured kernel clock is 150 MHz on both cards: 150,000 kHz on the U50, and 150 MHz from the U280 shell's frequency counter.
- **Flow** ([`flow/fpga/README.md`](../flow/fpga/README.md)):
  - `package_xo.tcl` builds the `.xo`.
  - `link.sh` runs `v++ -l` and reads the frequency back from the `.xclbin`. It fails on any clock-moving `AUTO-FREQ-SCALING` rule, and `LINK_RECHECK=1` re-checks a finished link.
  - `program.sh --check/--go/--status` gates loading on the shell UUID, an idle device, no concurrent loader and temperature.
  - `boards.sh` holds the per-card values (`TNCA_BOARD=u50|u280`).
- **Runbook:** [`r38_board_bringup.md`](r38_board_bringup.md).

| | Alveo U50 | Alveo U280 |
|---|---|---|
| Part / platform | xcu50-fsvh2104-2-e / `xilinx_u50_gen3x16_xdma_5_202210_1` | xcu280-fsvh2892-2L-e / `xilinx_u280_gen3x16_xdma_1_202211_1` |
| Timing at 150 MHz | WNS +0.055 ns, WHS +0.009 ns | Link 1 (placer's choice): WNS −1.919 ns on one AXI-Lite path crossing SLR0→SLR1. Link 2 (kernel pinned to SLR1): **WNS +0.016 ns**, kernel clock domain +4.229 ns, 0 failing endpoints |
| Utilisation | Not recorded | Whole device including the XRT shell: 24.1 % LUT, 12.8 % FF, 18.2 % BRAM tiles. Not broken down by kernel. |
| Loaded | 2026-09-26: an image that predates R38 and lacks `PERF_READ` | 2026-10-08: `tnca_u280.xclbin`, sha256 `f8717adc…a77a52bec` |
| Board evidence | read/ping/run (243 cycles, 72 instructions) | Conformance 5/5, profiler (full run), ArchBench tiny + small, multi-step generation (§10) |
| Temperature | FPGA 91–95 °C during runs | FPGA 32 °C, HBM 26–27 °C at load |

## 10. Measured behaviour

### 10.1 Board correctness and cycle fidelity (U280, 2026-10-08)

The acceptance ladder of [`r38_board_bringup.md`](r38_board_bringup.md) §2C was run on the U280:

| Step | Result |
|---|---|
| B1 conformance (`fpga-u280`, `--require-measured`) | 5/5 cases pass: lifecycle, transfers, compatibility, a successful launch (23 cycles, oracle value written) and a fault (code 0x40 at its PC, session still usable afterwards) |
| B2 profiler | Identities, monotonicity and no-perturbation hold through `PERF_READ` (quick run, then the full acceptance run) |
| B3 ArchBench tiny | 27/27 pass, every one within tolerance, counters recorded on all 27; 6.3 minutes |
| B3 ArchBench small | 25/25 pass, including LLaMA prefill (S = 64, 155.6 M cycles = 4.15 s) and decode (P = 128, 43.9 M cycles = 1.17 s); 7.7 minutes |
| B4 cycle fidelity | All 27 tiny points (132 launches) have device cycles identical to the `core-vrf-sync-v1` RTL simulator, launch by launch. The small tier has no RTL reference: it is too slow to simulate. |
| B5 cross-platform scoring | See §10.3 |
| B6 multi-step LLaMA generation | `python -m archbench.generate --tier small --new 8`: a 64-token prefill, then 7 decode steps, each on the KV cache the device produced. All 8 steps are within tolerance of an FP32 reference given the same inputs. The generated token ids equal those of a generation run wholly in FP32 (weights are random, so the tokens carry no meaning). Device time at 37.5 MHz: prefill 4.16 s (15.4 tokens/s); decode 0.80–1.04 s per token (1.01 tokens/s). Host wall time was 327 s: compiling each step's new shapes, plus 194 s in launches (transfers and waiting). |

Small LLaMA generation first needed one runtime fix: the device heap allocator did not merge freed blocks, so a long graph fragmented the heap. See [`compiler_overview.md`](compiler_overview.md) §8.3. Tiny-tier cycles stayed identical to the simulator after the fix.

### 10.2 Where the cycles go

Board counters for `model.llama_decode/tiny` (42 launches, 3.54 M cycles):

| Counter | Cycles | Share of cycles |
|---|---|---|
| Issued | 0.93 M | IPC 0.26 |
| Stall: register hazard | 1.06 M | 30 % |
| Stall: LSU | 0.45 M | 13 % |
| Stall: VRF bank conflicts | 0.41 M | 12 % |
| Stall: transfer engine | 68 k | 2 % |
| Stall: fence | 36 k | 1 % |
| Stall: FPU | 7.6 k | 0.2 % |
| Stall: matrix | 0 | 0 % |
| Memory beats (read / write) | 459 k / 302 k | |

Across the tiny tier the same pattern holds (`doc/r38_stage1_status.md`). Dependent instruction chains with no scheduling, a single 8-byte memory port, and repacking through memory dominate. The matrix unit is at most about 0.5 % busy even in GEMM, because tiles are at most 16 wide (§12) and operands arrive through the LSU.

### 10.3 Against a GPU

ArchBench compares platforms on the same PyTorch programs, each compiled by its own `torch.compile` backend. The GPU is an NVIDIA GB10 timed with CUPTI from cold caches, using CUDA-graph replay. TNCA is measured on the U280 at 37.5 MHz. Strict bounds are computed per platform from spec peaks (archbench/PROTOCOL.md).

| Tiers tiny + small (52 points) | TNCA U280 | GB10 |
|---|---|---|
| Pass | 52/52 | 52/52 |
| E (harmonic mean of η) | 0.0072 | 0.0051 |
| A (arithmetic mean of η) | 0.021 | 0.041 |
| Time ratio S = T_GPU / T_TNCA per point | 2 × 10⁻⁵ to 6 × 10⁻³ | |

η measures how close each platform comes to its own bound, not speed. At these sizes a GB10 kernel is dominated by launch latency (about 1.5 µs). One TNCA core at 37.5 MHz is 180× to 50,000× slower per point in seconds. The comparison is meaningful at the level of efficiency and attribution, which is what ArchBench is for (roadmap §11). Matching the GPU's absolute speed is not the success criterion.

## 11. Verification and evidence

- **RTL:** unit benches per functional unit (`tb/core`, `tb/nuca`, `tb/noc`, `tb/xls`), 46 core test-program sources (`tb/core/progs/`), the 2×2 system bench, the FPGA tops in simulation (`r36-fpga-tests`, 8/8 including `PERF_READ`), lint and yosys synthesis checks.
- **Compiler on RTL:** all 45 original TileBench operators, 149 cases at O0/O1/O2 (447/447), with numerics checked on the RTL. Details in [`compiler_overview.md`](compiler_overview.md) §12.
- **Backends:** `tools/r36/conformance.py`. Hardware backends count as not-applicable unless given `--xclbin`, and `--require-measured` turns that into a failure.
- **Evidence registry:** [`spec/r38/evidence_registry.json`](../spec/r38/evidence_registry.json) holds 21 claims, each with its evidence level (documented → implemented → functionally-validated → cycle-validated → synthesized → board-measured). Stage 1 forbids `board-measured`. Promoting the board claims is a Stage 2 entry item (§13).
- **CI** (`.github/workflows/ci.yml`):
  - `check`: lint, unit, core, the 2×2 system, the compiler, profiles, R3 models and synthesis.
  - `triton`: the real Triton frontend, ABI and runtime tests, package install.
  - `tilebench`: O0/O1/O2.
  - Neither the R38 PyTorch tests nor ArchBench nor the board run in CI. They need PyTorch, long RTL runs, or hardware.

## 12. Known limitations

1. **One core, serial grid.** A grid's programs run one after another. The 2×2 system is not reachable from the runtime.
2. **Memory capacity.** 512 KiB of on-chip memory and a 384 KiB heap. Llama-3.2-1B-sized work (the large tier) cannot run. The U280's 8 GB of HBM2 and 270 Mb (33.75 MB) of URAM are unused.
3. **Host round trips.** Every launch uploads its argument storages and downloads them back, one 64-bit word per register access. On the board this host time dwarfs device time: tiny LLaMA decode spends 1.9 s in transfers for 0.094 s of device time.
4. **Throughput.** Single issue, one 8 B/cycle port, 16 MAC/cycle matrix, a one-lane SFU, and no instruction scheduling in the compiler. η is about 1 % on models.
5. **Narrow tiles.** At most 16 elements per block dimension (compiler limit, [`compiler_overview.md`](compiler_overview.md) §14). This caps matrix-unit use.
6. **Transfer engine.** A box row must be a whole number of 8-byte units (the compiler reads other boxes through the LSU), and a misaligned row costs two memory reads per unit. There is no transposed store, no TMA-style gather4/multicast, and no reduce-in-memory (descriptor atomics are per-element LSU atomics).
7. **Not a physical design.**
   - FPGA timing: the loaded image runs at 37.5 MHz. The R38 timing work makes 50 MHz the default for new builds; this is estimated out of context and not yet linked.
   - Otherwise there are only yosys structural proxies: no ASIC PPA, and no power measurement yet.
8. **Documentation drift.**
   - ISA text: atomics and register classes marked reserved; vector-register base.
   - `spec/r2_transport_v1.json` lacks op 7.
   - `rtl/fpga/tnca_u50_top.v` still says nothing has run on a board.

## 13. Next steps

Ordered by what the R38 plan needs first. Each item names the decision or evidence it produces.

### 13.1 Close Stage 1 and record the board

- **Rebuild the board image for the rank-5 transfer engine. Done, 2026-10-09.** The U280 now runs a 100 MHz undivided image of r38-dco `8d8e4e9`, which has the rank-5 engine and `t.st.async` (`evidence/r38/u280_100mhz/`). The image loaded on 2026-10-08 executed rank 1–2 only and must not run descriptor kernels; it is no longer loaded. On the new image, through `fpga-u280`:
  - `tools/r38/tdesc_tests.py` passes 10/10, including rank-3 transfers both ways and the rank-5 zero fill;
  - `tools/r38/torch_tests.py` with `TNCA_TENSOR_DESCRIPTORS=1` passes 24/24 (R39-00).

- **Record the board as evidence.**
  - The U280 results (B1–B6) are in the runbook (§5 there) and in `doc/r38_stage1_status.md`.
  - Promote the `fpga-board` claim in the evidence registry from `documented`, using the Stage 2 levels.
  - Record per-kernel utilisation from the U280 link reports.
- **Finish the open Stage 1 items:** S1-05 item 7 (degraded RTL builds), S1-08 ablations, S1-09 L2 promotion run, S1-11 related work.
- **Fix the drift listed in §12.8.** Bump the transport spec to 1.1 with op 7, and revise the ISA text to v0.2.

### 13.1a The board clock

The first timing pass is done ([`r38_board_bringup.md`](r38_board_bringup.md) §6).

- **What changed:** the matrix unit's accumulate loop is cut. The FMA is generated by XLS, and the latency-2 instance is a drop-in. The tensor engine's window check is precomputed when the descriptor loads.
- **Result:** with cycle counts unchanged, new builds default to 50 MHz.
- **Next:**
  - Link once at /3 to confirm.
  - Then decide on `FP_LAT=3`, which is what 75 MHz needs. It is a cycle change, so it means a re-frozen baseline and a design-change record.

### 13.2 Memory: the next hardware decision

The board can now run every small-tier model. The next workloads (larger LLaMA shapes, multi-step generation with longer caches, the large tier) are bound by memory capacity. There are two paths ([`r38_board_bringup.md`](r38_board_bringup.md) §3):

| Option | What it takes | Gives |
|---|---|---|
| URAM | Change `WORDS`/`ADDR_W` and the memory inference; redo timing | Up to 22.5 MB (U50) / 33.75 MB (U280) of on-chip memory, same latency model |
| HBM | An AXI master and a DMA path in the device; a latency model the simulator does not have today | GiB-scale memory; the system's real DRAM path |

**This needs a decision from the project lead.** URAM is a bounded change that keeps board/simulator cycle identity. HBM is the path to the large tier and to a meaningful memory hierarchy.

### 13.3 Device-resident tensors (PyTorch P2)

Keeping tensors on the device between launches removes the per-launch round trips (§12.3). It turns host time into something comparable with device time, and it is a prerequisite for a credible tokens/s figure (roadmap §12.4 P4–P5). This is mostly software (a PrivateUse1 device in PyTorch, see [`compiler_overview.md`](compiler_overview.md) §15). It depends on §13.2 for anything beyond toy models.

### 13.4 Throughput, guided by the counters

The counters say where to start. Each change below is an R38 cross-layer case: the profiler attributes the cost, the cost model predicts the gain, and the RTL measures it.

- **Register hazards (30 % on LLaMA decode).** The first lever is compiler scheduling, which costs no hardware. After that: dual issue (the decoder already carries FU classes) and forwarding.
- **Memory port (13 % LSU stalls plus serialised transfers).** A wider data port, or separate ports for the transfer engine and the LSU.
- **Matrix utilisation.** This needs wider tiles from the compiler first. A larger matrix unit is pointless until operands are delivered by the transfer engine at tile width.
- **VRF bank conflicts (12 %).** Bank-aware register allocation (`bankcost.py` exists) versus more banks or ports.

### 13.5 Toward the design: multiple cores and the cache

- Bring the 2×2 system into the runtime: grid programs distributed across tiles, and per-tile counters.
- The draft's Stage 4 cache ideas, one at a time with an ablation each. TensorMap metadata to the NUCA, dead hints and coordinated replacement/bypass are now in, as DCO ([`dco.md`](dco.md); measured there per policy on the 2×2 RTL). What remains is MSHR-aware arbitration and request admission. The DCO results also need a faster system simulator, since the 2×2 RTL runs at about 300 cycles/s under iverilog, and a full-size last level.
- A 4×4 configuration needs the parameterisation fixed first (audit F05).

### 13.6 Physical evidence

- Power and energy on the board for the P5 demo (`xbutil examine -r electrical` reports the card's power rails).
- An ASIC-style synthesis estimate of one core if the paper needs area/power claims beyond structural proxies.

## 14. Source map

| Area | Where |
|---|---|
| Core | `rtl/core/tnca_core.v` and units in `rtl/core/` |
| Device top | `rtl/tnca_shell.v` |
| Multi-tile system | `rtl/tnca_top.v`, `rtl/tile/`, `rtl/nuca/`, `rtl/noc/` |
| DCO (cache orchestration) | `rtl/nuca/tnca_tmu.v`, `rtl/nuca/tnca_nuca_slice.v`, `compiler/python/tnca_cc/dco.py`, `tools/r38/dco_bench.py`; [`dco.md`](dco.md) |
| Generated datapaths | `rtl/gen/` (from `xls/`) |
| FPGA | `rtl/fpga/`, `rtl/board/`, `flow/fpga/` |
| Benches | `tb/` (`tb/device/tnca_device.v` is the simulator device behind the runtime) |
| ISA | `doc/isa/TNCA-ISA-v0.1.md`, `isa/tnca-isa-v0.1.json`, `tools/asm/tnca_isa.py`, `tools/isa/` |
| Profiles and targets | `spec/r1_profiles.json`, `compiler/targets/*.json` |
| Counters | `spec/r38/counters.json`, `tools/r38/gen_counters.py`, `rtl/common/tnca_perf_*.vh` |
| Runtime and backends | `python/tnca/` ([`compiler_overview.md`](compiler_overview.md) §8) |
| Benchmarks | `archbench/` (PyTorch level), `tools/r38/archbench/` (scoring) |
| Plans and status | [`r38_isca_roadmap.md`](r38_isca_roadmap.md), [`r38_stage1_tasks_and_acceptance.md`](r38_stage1_tasks_and_acceptance.md), [`r38_stage1_status.md`](r38_stage1_status.md), [`r38_board_bringup.md`](r38_board_bringup.md) |
| History | `doc/phase*_tasks_and_acceptance.md`, `doc/r1_status.md` … `doc/r37_validation.md`; the original design in `doc/design/` |
