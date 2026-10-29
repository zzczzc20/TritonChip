# Which G2 cut makes the five points slower than run 1

**Domain: RTL simulation** of the board top (`tnca_multicore_hbm_xo` in the board bench) with the testbench memory model. These are not board measurements.
- Points: 64 KiB registered points. The default compiler was used (`TNCA_TILED_POINTWISE` 0).
- Tool: `tools/r41/link_budget_rtl.py`.
- Every point passed its numerical check (`w*.log`).

Each arm applies **one** of G2's approved cuts to the run-1 integration image (`evidence/r41/stage2/integration/image.tcl`). The images are G2's own single-cut screens in `resources/screens/`. The contexts arm uses the profile that matches its image (`core-vrf-sync-v5-x8-g2`); every other arm uses the run-1 profile. The run-1 image was rerun in the same tree as the reference. The last row is G2's combined image, from `../rtl_s4/controls/`.

Table (`table.md`; `python3 tools/r41/g4_g2_cuts_report.py evidence/r41/hbm_bw/g4/g2_cuts`). Each cell is device cycles, with run-1 cycles divided by the arm's cycles in brackets (below 1 is slower):

| Image (one cut on the run-1 image) | add fp16 strided | clone fp16 strided | exp fp16 | silu fp16 | sum fp16 | geometric mean |
|---|---:|---:|---:|---:|---:|---:|
| run-1 image (rerun) | 152,821 (1.00x) | 104,352 (1.00x) | 104,651 (1.00x) | 132,841 (1.00x) | 33,641 (1.00x) | 1.000 |
| **contexts 4→2** (`NUM_RCTX 2, RCTX_W 1, PHYS_VREGS 128, PHYS_W 7`) | 167,977 (0.91x) | 110,517 (0.94x) | 131,368 (0.80x) | 163,930 (0.81x) | 54,462 (0.62x) | **0.807** |
| barriers 32→8 with 38-bit epochs | 153,102 (1.00x) | 104,602 (1.00x) | 104,625 (1.00x) | 132,923 (1.00x) | 33,733 (1.00x) | 0.999 |
| 38-bit epochs alone | 153,084 (1.00x) | 104,352 (1.00x) | 104,382 (1.00x) | 132,649 (1.00x) | 33,626 (1.00x) | 1.001 |
| 35-bit home address | 152,803 (1.00x) | 104,161 (1.00x) | 104,726 (1.00x) | 132,937 (1.00x) | 33,584 (1.00x) | 1.000 |
| `NUCA_SHARED_DATA_READ 2` | 153,084 (1.00x) | 104,617 (1.00x) | 104,582 (1.00x) | 132,882 (1.00x) | 33,951 (0.99x) | 0.997 |
| `AXI_LINE_LUTRAM 1` | 152,524 (1.00x) | 104,372 (1.00x) | 104,525 (1.00x) | 133,042 (1.00x) | 34,151 (0.99x) | 0.997 |
| all G2 cuts (`image_g2.tcl`) | 168,022 (0.91x) | 109,940 (0.95x) | 131,383 (0.80x) | 163,962 (0.81x) | 54,285 (0.62x) | 0.808 |

## Result
**The contexts cut (4 → 2 role contexts per core) accounts for the whole slowdown.** It makes the five points 6–38 % slower: sum 0.62×, exp 0.80×, silu 0.81×, strided add 0.91×, strided clone 0.94×. Its geometric mean of 0.807 equals that of all G2 cuts together (0.808). The other five cuts change no point by more than 1 % (0.997–1.001).

**Mechanism** (rtl counters, run 1 → contexts 4→2):
- Fewer programs run concurrently on a core, so fewer memory requests are in flight. The mean outstanding requests at the core port fall: sum 4.14 → 2.47, silu 3.64 → 2.94, exp 4.73 → 3.79.
- The latency of each request does not change (sum 50 → 49 cycles, exp 73 → 73).
- The core is idle more: IPC falls (sum 0.497 → 0.307, exp 0.248 → 0.197, silu 0.293 → 0.237), while LSU stalls drop (sum 0.217 → 0.036). The two remaining contexts do not saturate the memory path; they wait.
- This is a loss of latency hiding, not a slower path. A program class with little memory-level parallelism of its own (a reduction, the SFU activations) loses the most.

This is a measured slowdown caused by an approved capability cut. Under discipline §2.4.2 it is a trade-off for the author: the cut saves 18,476 kernel LUTs, and in the tile-0 screen 41,878 → 41,175 CLBs, 76.20 % → 74.92 % of SLR0 (RESOURCES.md §9), and costs these points 6–38 % (rtl).

Only the five points that were slower under G2 were measured. The cut's effect on the other registered points has not been measured.
