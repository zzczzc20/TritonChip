# G4 run-2 candidate image (frozen, not linked)

Status: **frozen 2026-10-29 at commit `1f0a5aff`** (RTL and compiler as gated at `55c71e0e`). It is **not linked: the lead starts the link.** Nothing here was loaded onto a card. [`doc/engineering_discipline.md`](../../../../../doc/engineering_discipline.md) is binding.

## What the image is
`image.tcl` is G2's combined image with four contexts, the one-lane path of [`G4_DESIGN.md`](../../G4_DESIGN.md) §8, and the run-2 steps of §9.6. Its profile is `core-vrf-sync-v5-x8-g2c4-dp` (MAC_DP 1).

Decisions it carries:
- **Four contexts.** Source: the author, in the lead's chat, 2026-10-29. The 4→2 cut was withdrawn because of its measured slowdown.
- **The line-wide tensor engine** with the transposed-load line buffer (`TENSOR_LINE 1`). For the FP8 weight stream, the Hub FP8 matmul's 512-byte transfer goes from 414 to 174 cycles at latency 60 (rtl).
- **Chained home fills** (`HOME_LINE_PIPE 2`): a read miss costs one cycle per line instead of two (rtl, one slice).
- **64 lines in flight** per core and per home (`LINE_SLOTS 64`, `HOME_MSHR_N 64`).
- **512-bit outlets with 32 IDs.** The lead's decision for SLR0: 64 IDs did not fit.
- **`LSU_INFLIGHT 1`.** The trade-off was approved by the author (source: the author, in the lead's chat, 2026-10-29):
  - rtl: copy fp32 mode 3 is +9.8 %;
  - TLM upper bound: decode step +3.3 %, prefill +2.3 %, attention prefill +9 to +20 %;
  - Hub FP8 kernels are unaffected.
  - Details are in RESOURCES.md §11.
- **G2's FMA rebuild and shared MAC lane** (`FMA_IMPL 1`, `MAC_DP 1`, `MAC_CUTS 29`), **K8's registered matrix operand capture** (`MTX_VRF_REG 1`) and `HBM_STRIPED`.

Every new option is default-off in the RTL. The compiler fences TensorMap snapshots only for the line-engine profiles (`compiler.snapshot_fence`). The default output of all 113 original points is byte-identical to `3a37b7e6`.

## Evidence before the link

| Check | Domain | Result | Where |
|---|---|---|---|
| Kernel synthesis | screen | 812,545 LUTs | `../screens/frozen_0802ae88/` |
| Tile 0 in SLR0, out of context, 10 ns | screen | 252,228 LUTs, 43,345 CLBs (78.9 %), WNS +0.600 ns | same |
| Tile 1 in SLR1 | screen | 248,099 LUTs, 42,369 CLBs, WNS +0.641 ns | same |
| Whole-SLR0 estimate | estimate | 95.5 ± 2.5 % CLB (run 1: 96.74 %) | G4_DESIGN §9.6, RESOURCES §11 |
| XO packaging, frozen image | packaging | exit 0, no critical warnings; kernel.xml and IP m_axi ports at 512 bits | `package_dryrun_frozen.log` |
| Board-top geometry, frozen image without striping | rtl | pass (code upload, two grids, memory controls, coherent window, handoff) | `geometry_s7.log`, gate below |
| Default-change audit, 113 points vs `3a37b7e6` | compile | 113/113 unchanged | gate `~/r41g4_gate4` (GB10) |
| Line-engine miter (fp32/fp16/fp8, transposed loads), X8 perf and store-race tests | rtl | 18/18; XPERF; XRACE 1 and 2 | `../tte_rtl/` |
| Home regressions, plain and chained | rtl | 189/189 | `../home_chain/` |

Run 1 lost 0.9–1.0 ns from tile screen to full image at this packing. The tile's +0.60 ns therefore predicts a small negative or near-zero margin at 100 MHz (estimate).

## Fallback
`image_fallback_axi256.tcl` is the same image with 256-bit outlets (32 IDs, no platform width converters). It saves about 440 SLR0 CLBs (94.7 ± 2.5 %) and halves the outlet bandwidth. The tiles are unchanged, so the tile screens above apply.

## How to link (the lead)
On the x86 host, from the archive of the frozen commit (`~/r41res/src_1f0a5aff.tgz`, sha256 `435831da62fe43a3c0cd66c1fc29c792dc91f98390403b11111700f43057482a`), with the Vitis HLS and `unzip` requirements found by the x86 dry run ([`../../x86_link_dryrun/README.md`](../../x86_link_dryrun/README.md)):

```sh
export RUN2_SOURCE=$HOME/r41run2/src_1f0a5aff FRESH_BUILD_ROOT=$HOME/r41run2/build_1f0a5aff   # must not exist
mkdir -p "$RUN2_SOURCE" && tar xzf ~/r41res/src_1f0a5aff.tgz -C "$RUN2_SOURCE"
source ~/x86_link_env.sh
export XILINX_HLS="$HOME/Xilinx/Vitis_HLS/2023.2"
export PATH="$HOME/r41-codex-t9d-bin:$XILINX_HLS/bin:$PATH"
cd "$RUN2_SOURCE"
setsid nohup nice -n 5 bash flow/fpga/run_r40_attempt01.sh "$FRESH_BUILD_ROOT" "$RUN2_SOURCE/evidence/r41/hbm_bw/g4/run2/image.tcl" > /dev/null 2>&1 < /dev/null &
```

**Expectation (estimate):**
- Duration: 5–7 h. Run 1 took about 3 h 40 on the 24-CPU FPGA host.
- Memory: peak 30–35 GB, so swap may be used.
- No other Vivado job should run on the host during the link.
- One link counts against the Stage 2 budget of five full runs.

**If placement fails, check first:**
- the SLR0 utilisation per SLR;
- what was placed in SLR0: tile 0, three outlets, three width converters, the HBM subsystem, the DCO replica;
- the placement pre-hook's tile pblocks;
- congestion near the HBM column.

**If routing fails or timing misses, check first:**
- tile 0's worst paths. In the screens these are the hybrid LSU's done → out_vector and the tensor engine → LSU word unit.
- SLR crossings of the outlets and converters.
