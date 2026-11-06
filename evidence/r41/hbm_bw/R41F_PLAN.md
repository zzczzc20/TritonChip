# R41-F: the final three-core image (high-bandwidth stream compute) and board chat

> **Stopped by the user on 2026-10-28.** The stream-compute-unit direction below (workstreams A, B and C) was stopped before any design work landed. It is kept as a record only. The user found a fixed-function unit next to the HBM too narrow for general PyTorch lowering. Workstream D (complete Qwen models and chat on the run-1 image) continues on `r41`.

Owner: the lead (Claude), from 2026-10-28. Codex and Kimi are out of quota; the work runs in Claude subagents. This plan supersedes the open-ended V2/V3 exploration in `STATUS.md`, whose evidence it builds on.

## 1. Goals
1. **R41 delivery:** the contract `doc/r41_tasks_and_acceptance.md` closed, or each remaining item shown failed or blocked with its reason, on a final image.
2. **Chat on the board:** run Qwen3-VL-2B-FP8 and Qwen3.5-4B-FP8 from a plain Python/PyTorch script (`tools/models/chat.py`) on the U280 and talk to them.
3. **High bandwidth, cashed in as operator speed:** a memory system designed for 100 GB/s or more, and on the board several-times faster standard torch element-wise operators (gate: at least 5x for `add`, `mul`, `copy` at large sizes against the run-1 100 MHz image), plus the weight-streaming bandwidth that makes decode interactive.

## 2. What the evidence already settles
- **The bottleneck is the core-side path,** not HBM: three 64-bit core ports cap the ordinary path at 2.4 GB/s; the measured run-1 large `add` is 0.161 GB/s of logical traffic (`evidence/r41/stage2/roofline/points.json`). Widening the ordinary core path does not scale: V3 (eight 64-byte services per core) reaches only 3.98 GB/s in the model, and the general 8-unit LSU bank does not fit (440,741 LUTs per core).
- **A native stream path does scale:** the compact ordered-burst lane (`rtl/fpga/tnca_stream_axi.v`; 1,560 LUTs, 7.5 BRAM tiles, routed +5.596 ns alone) with 16-beat bursts and eight outstanding bursts is modelled at 222.6 GB/s copy, 111.2 GB/s fill and 162.3 GB/s read-reduce for 32 lanes at 100 MHz, within 0.7 to 6.4 % of the single-lane RTL (`tools/r41/stream_bandwidth_model.py`).
- **The physical constraint decides the geometry:** SLR0 (next to the HBM) is at 96.74 % CLB packing in run 1 and its SLR0-SLR1 crossing use is 14,263 of 23,040 SLLs; the V2 core-plus-32-stream screen failed placement (tile 1's SLR0 pblock needed 18,918 CLBs, 2,776 available).
- Decode is weight streaming: an M=1 linear reads every weight once per token. 2 GB of FP8 weights at 0.16 GB/s is about 12 s per token; at 20 GB/s about 0.1 s; at 100 GB/s about 20 ms.

## 3. The R41-F architecture
- Keep the run-1 three-core design unchanged as the general path (instruction semantics, numerics and the 113 original points untouched; the default-off hybrid LSU work may be enabled only where it passes regression and screens).
- Add a **stream compute unit (SCU)**: N compact stream lanes, each owning a set of HBM pseudo-channels, with a 4 KiB-page striped logical-to-physical layout over the pseudo-channels (advertised as an image capability and implemented identically by the core outlets and host DMA), a command/control interface on its own CU or register window, and per-lane compute:
  - copy, fill, and element-wise binary operations (`add`, `sub`, `mul`) on fp32, fp16 and bf16, contiguous and aligned, with tails and the other cases left to the core path;
  - unary activations (`silu`, `gelu`) built from the **same XLS-generated arithmetic units as the core path**, so results are bit-identical to what the core computes today;
  - a **GEMV mode** for M=1 linears: stream BF16 or FP8 (e4m3, 128x128 block-scaled) weights, multiply by a resident activation vector, accumulate in fp32 per block, apply scales; numerics judged against the CPU references within the tolerances already derived (`tools/models/fp8_blockwise_ref.py`, chat teacher tolerances).
- Runtime: the `tnca` torch device dispatches eligible operations to the SCU (same session lock, drain the cores, cache writeback or invalidate of the touched logical ranges, run, verify completion and fault state), and everything else stays on the cores. No CPU numerical fallback.

## 4. Workstreams (parallel)
| ID | Owner | Scope | Exit evidence |
|---|---|---|---|
| A | arch/physical subagent, worktree `TritonChip-hbw-arch` | Choose N, lane width and placement (SLR0 vs crossing), pseudo-channel mapping, SLL and CLB budget; replicated-lane screens with the real three-tile occupancy; floorplan constraints; the integrated top; free kernel and tile screens at 10 ns | A frozen geometry with routed screens of the integrated design that predict 100 MHz, and a written area/SLL budget |
| B | RTL subagent, worktree `TritonChip-hbw-rtl` | SCU lane compute: element-wise binary, activations from the XLS units, GEMV with FP8 block scales and BF16; command/control; faults; exact-oracle and randomized tests; single-lane and N-lane unit screens | Bit-exact (element-wise, activations) and reference-tolerance (GEMV) tests passing on RTL simulation; unit screens with margin |
| C | software subagent, worktree `TritonChip-hbw-sw` | Runtime dispatch from torch ops (elementwise, copy, M=1 `linear`/`_scaled_mm`/Hub FP8 kernel route) to the SCU; striped layout in the allocator and DMA; SystemC/TLM model of the SCU calibrated to the RTL; projected operator speedups and decode tokens/s; RTL co-simulation of standard torch points | Standard torch points correct through the SCU on RTL simulation and the model, with projections that meet the gates |
| D | full-model subagent, worktree `TritonChip-p3` (branch `r41`) | P3: the complete Qwen models on the board on the run-1 image through the standard path, plus the chat driver; the board chat queue (`spec/r41/chat/`) results | Full-depth generation on the board with teacher-forced agreement and a measured tokens/s, or the exact blockers |
| E | lead | Integrate A, B and C into one tree, run the full link on the x86 host (run 2 of 5, pre-authorized), get the user's approval to load it, run the board acceptance batch, the chat demo, and the final reports | R41-F on the board, the operator speedups and decode rate measured, `doc/r41_report.md` final |

## 5. Rules for every workstream
- Real runs only: every number comes from a logged run whose raw output is kept; never type a measurement into code or a document; no mock devices standing in for hardware; failures are preserved and stay failures until fixed and re-measured. (An earlier session's typed-in numbers were reverted.)
- The cap of five full runs total (one used) stands; only the lead starts full runs. Free screens: the x86 host (at most two Vivado jobs, `source ~/x86_link_env.sh`) and the FPGA host (through `~/r41s2_tools/vivado_slot.sh`).
- The shared U280: only the lead loads images and starts board queues; workstreams write manifests and waiting launch scripts and give the exact command.
- Branches: workstreams A to C push to `origin/r41-hbm-bw`, D pushes to `origin/r41`, fast-forward only (`git fetch` and merge first); no new remote branches. New features are default-off until the integrated image selects them.
- Server access, limits and card rules: `evidence/r41/SERVERS.md`.
