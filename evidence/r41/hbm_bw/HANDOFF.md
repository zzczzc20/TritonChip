# Handoff: raise the effective HBM bandwidth by an order of magnitude (target: at least 100 GB/s)

This is a long-running task. It ends when a new image candidate that reaches the bandwidth target has been built, passes timing, and has been measured on the board. Keep working toward that goal. Push every result as soon as it exists.

## 1. Why

The roofline of the 113-point ArchBench suite (`evidence/r41/stage2/roofline/README.md`) shows that TNCA sits about 29× below the roofline of its own three-core hardware, and that the bottleneck is the memory path on the core side, not the HBM:

| Memory ceiling at 100 MHz | Bandwidth |
|---|---:|
| One core's memory port (8 B/cycle, `rtl/core/tnca_mem_mux.v`) | 0.8 GB/s |
| Three cores' ports | 2.4 GB/s |
| Three AXI outlets of 256 bit (`tnca_line_axi`) | 9.6 GB/s |
| U280 HBM2, theoretical (32 pseudo-channels × 14.4 GB/s) | 460 GB/s |

The current image (`r41s2-run1`, UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`, 100 MHz) has three cores, `MEM_PORTS=3`, three AXI outlets with `AXI_DATA_W=256`, and each outlet can reach all 32 HBM pseudo-channels (`--connectivity.sp tnca_u50_1.hbm{0,1,2}:HBM[0:31]`). The R40 model attribution (`evidence/r40/v3/model_attribution/README.md`) found that streaming is limited by the NUCA home-slice engine and the per-core memory path, and that adding outlets alone changed nothing (`op.copy/hbm`, ports 1→3: 1.00×; the outlet was busy 12 % of the time).

**The goal is at least 100 GB/s of effective streaming bandwidth on the board**, which is an order of magnitude over the current ceiling. That needs more memory channels, a wider and deeper path from the cores to the outlets, and no single-engine bottleneck in between. Everything is open: the number of outlets and pseudo-channels, the outlet width and outstanding depth, the home-slice engine, the NoC, the core ports, the address interleaving across pseudo-channels, the transfer engine, and whether data should bypass the NUCA for streams.

## 2. What to deliver

1. A written analysis of where the bytes are lost between a core and the HBM, with evidence: board counters (AXI read and write counters, outlet wait cycles, home-slice queues), the SystemC/TLM model, and RTL simulation. Quantify each stage's ceiling.
2. A design for the next image that removes the bottlenecks, with an estimate of the bandwidth, the area (SLR0 CLB packing is already 96.7 %) and the timing risk. Compare options before choosing. Justify the choice with numbers.
3. The implementation, with functional regression. It must keep the numeric semantics of every existing instruction and the 113 original points' behaviour.
4. Screens: unit and tile out-of-context synthesis and place-and-route (these are free), then full runs (see the budget below).
5. Board measurements of the new image: a streaming microbenchmark (copy, read-reduce, write) at several sizes and core counts, plus the ArchBench suite and a re-drawn roofline (`tools/performance_harness/roofline_report.py`).
6. A status file, `evidence/r41/hbm_bw/STATUS.md`, updated at every milestone (what was learned, what is running, what is next).

## 3. Rules

- **Branch and worktree:** your worktree is `/home/ubuntu/TritonChip-hbmbw`, on branch `r41-hbm-bw` (from `origin/r41`). Work only there. Another Codex session works in `/home/ubuntu/TritonChip` on branch `r41`; do not touch that folder. The lead (Claude) works in `/home/ubuntu/TritonChip-main`.
- **Commit and push at every milestone**, immediately: `git push origin r41-hbm-bw`. This machine has no swap and little memory, and unpushed work can be lost. Never force-push, amend or rewrite history. Commit as `git -c user.name=zzczzc20 -c user.email=zhouzc2000@foxmail.com commit`, with a message ending in a blank line and `Co-Authored-By: Codex <noreply@openai.com>`. Stage files by explicit path. When a result is verified, merge it into `r41` (`git merge origin/r41` first, fix conflicts, then push `r41-hbm-bw`; ask the lead to merge, or push to `r41` yourself only as a fast-forward). The number of active R41 branches must stay at two or fewer: `r41` and `r41-hbm-bw`. Do not create more.
- **Documents are professional English.** The documentation graph is `doc/docmap.toml`; `python -m tools.docmaint check` must pass, and new Markdown documents get an entry.
- **Never write credentials into any repository file.** Access goes through the helper scripts below.
- **Frozen Stage 1 tree:** the Stage 1 product tree is commit `25094197f6e6` (software tree `files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`). Do not change what Stage 1 acceptance binds to.
- **Honest reporting:** failures stay failures, and a model prediction never stands in for a measurement. Prefer board counters, Proton and the SystemC/TLM model over long RTL simulations for performance questions.

## 4. The three servers

All access goes through helper scripts. They hold the credentials outside the repository, so you never need to type a password. Set once:

```sh
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
```

Do not copy the contents of these scripts into any file, and do not print them.

| Server | Command | What it is for |
|---|---|---|
| **GB10** (build and simulation host, user `zhongchun`) | `$S/gb 'command'` | RTL simulation (iverilog, verilator, yosys: `source ~/tc_env/env.sh`), the SystemC/TLM model, Torch (`~/tc_env/torchenv/bin/python`, never pip install into it), XLS is not available (aarch64). Also the jump host for the FPGA host. |
| **FPGA host** `acf3019` (user `zzhouch`) | `$S/fp 'command'` | The shared Alveo U280, XRT 2.16, Vitis/Vivado 2023.2 at `/tools/Xilinx`, XLS at `~/tnca_xls/.xls/cur`. 24 cores and 62 GB shared with other users. Full v++ links run here. |
| **x86 Vivado host** (WSL, 16 cores, 23 GB RAM + 8 GB swap, user `vivado`) | `$S/vx 'command'` | CPU-only Vivado for unit and tile screens. Vivado: `source ~/Xilinx/Vivado/2023.2/settings64.sh`. xcu280 needs no license. No card. Not enough memory for a full link (about 31 GB peak). |

File transfer:
- To GB10: `rsync -a --delete --exclude .git --exclude /evidence --exclude /sim -e $S/gb_rsh <worktree>/ gb10:<dir>/`
- To the x86 host: `rsync -e $S/vx_rsh <files> vivado@143.89.78.138:<dir>/`
- To the FPGA host: stream through GB10 (`cat file | $S/fp 'cat > ~/file'`, or `scp` from GB10 to `zzhouch@acf3019.ece.ust.hk`).

How work is organised:
- **GB10:** limits for a shared host: at most 6 cores and 16 GB of RAM per job, and keep 40 GB available (`free -g`). Other agents also use it. Kill processes only by PID, never `pkill -f`.
- **Synthesis screens (free):** `flow/synth/vivado_ooc.sh`, `flow/fpga/synth_r40_multicore.tcl` (kernel synthesis) and `flow/fpga/screen_r41_tile.tcl` (tile place-and-route in one SLR at 10 ns). Examples with the exact method are in `evidence/r41/stage2/integration/README.md` and `evidence/r41/stage2/candidates/T2/README.md`. Run at most two Vivado jobs at once on the x86 host. On the FPGA host run Vivado through `~/r41s2_tools/vivado_slot.sh --min-gb 8 LOG -- command` (two slots and a memory guard).
- **Full image link:** only on the FPGA host. The recipe is in `evidence/r41/stage2/run1/README.md` (it took 9 h and peaked at 31 GB) and the flow scripts in `flow/fpga/` (`package_xo.tcl`, `link.sh`, `place_r40_multicore.tcl`). The image parameters of the current image are in `evidence/r41/stage2/integration/image.tcl`. Run detached under `nice`, and check `free -g` first: at least 38 GB must be available. Never kill another user's job.
- **Full-run budget:** the author allows at most 5 full synthesis + place-and-route runs in total for the whole R41 Stage 2. One is used (run 1, the current image). Unit, kernel and tile screens do not count. Ask the lead before starting a full run, and record each one as n/5 in its evidence.

## 5. The shared U280 card

The card is shared with other users (`eeyhui` and `cliufc` run long jobs). Rules that are not negotiable:
- Never load, reset or open the card unless it has had no context, no outstanding exec and no client process for 15 consecutive minutes (`/sys/bus/pci/devices/0000:04:00.1/kdsstat`), and never override anyone's run.
- Loading a **new** image needs explicit approval from the user, through the lead. Ask before you load. Images already approved: the 90 MHz B40 image (`a88ebdf5…`) and the 100 MHz image `30e79769…`.
- Use `flow/fpga/program.sh --check` and then `--go`. Always export `TNCA_BOARD=u280` and source `/tools/Xilinx/Vitis/2023.2/settings64.sh` and `/opt/xilinx/xrt/setup.sh` first; otherwise `program.sh` checks a U50 and fails.
- Board experiments run through `tools/performance_harness/experiment.py` manifests, which check the image UUID and the clock before and after each step. The pattern for a waiting queue script that loads only when the card is idle is `evidence/r41/stage1/launches/board_queue.sh`, and `evidence/r41/stage2/board/launches/` for the 100 MHz batch. The queues that exist now: a Stage 1 rerun and a 100 MHz endurance rerun are waiting on the FPGA host (`~/r41_followup_chain.sh`, `~/r41_chain2.sh`). Do not interfere with them; schedule yours after them, and chain through log markers.

## 6. Where to start

- Read `evidence/r41/stage2/roofline/README.md`, `evidence/r40/v3/model_attribution/README.md`, `evidence/r41/stage1/latency_boundaries.md`, `evidence/r41/stage1/memory_paths.md`, `evidence/r41/stage1/memory_service.md` and `evidence/r41/stage1/hbm_controller/README.md`.
- The RTL path to study: core LSU → `tnca_mem_mux.v` → the NoC → the NUCA home slice (`rtl/nuca/`) → `tnca_line_axi.v` (the outlets) → HBM. Also `rtl/fpga/tnca_multicore_axi.v` and `rtl/board/tnca_multicore_hbm_xo.v`.
- The SystemC/TLM model (`tools/performance_harness/systemc/`) has both a fast and a Ramulator2 DRAM path and the real HBM address map; use it to rank design options before touching RTL. It is uncalibrated in absolute terms (R40 holdout median error 21.5 %), so confirm on the board.
- Board counters: the AXI window counters (`R_AXI_RD`, `R_AXI_WR`, `R_AXI_LAT`, `R_AXI_RBEAT`) in `python/tnca/backends/u50.py`, and the outlet and home-slice counters in the R40 records.
- Measured baseline: `evidence/r41/stage2/board/results.md` and `evidence/r41/stage2/roofline/points.json`.
- The clock is 100 MHz; the U280 HBM2 runs at 450 MHz. Wider outlets, more outlets and deeper outstanding windows all cost area in SLR0, so use SLR1 and SLR2 where possible.
