# G4 run-2 candidate image (prepared, not linked)

Status: prepared by workstream G4, 2026-10-28. **Not linked: the lead starts the link.** Nothing here was loaded onto a card. [`doc/engineering_discipline.md`](../../../../../doc/engineering_discipline.md) is binding.

## What the image is
`image.tcl` is G2's combined image (`resources/screens/image_g2_all.tcl`, profile `core-vrf-sync-v5-x8-g2`) plus the one-lane path of [`G4_DESIGN.md`](../../G4_DESIGN.md) §8:

| Parameter | Run-1 image (RTL default where absent) | Candidate | What it does |
|---|---|---|---|
| `HYBRID_LSU` | 0 | 1 | line-wide core LSU path and 720-bit NoC lane |
| `HOME_BACKGROUND_READS` | 0 | 1 | the existing home slice issues MSHR memory reads outside its engine states; no effect while `HOME_LINE_PIPE` is on, because the pipe then takes every home request (kept so that the image matches s1–s4) |
| `HOME_LINE_WRITE_NOFILL` | 0 | 1 | a full-line store allocates without reading memory |
| `HOME_LINE_PIPE` | 0 | 1 | three-stage line home, one request per cycle when it does not stall |
| `HOME_MSHR_N` | 8 | 32 | lines in flight per home |
| `LINE_SLOTS` | 8 | 32 | line LSU slots and NoC tags per core |
| `AXI_OUT_N` / `AXI_ID_W` | 8 / 3 | 32 / 5 | outstanding lines per outlet (`AXI_ID_W` is derived) |
| `AXI_PIPELINED_ISSUE` | 0 | 1 | back-to-back AXI issue |
| `HBM_STRIPED` | 0 | 1 | 4 KiB pages striped over the 32 pseudo-channels (layout `pc32-page4k-v1`; needs the explicit 8 GiB allocation the runtime already enforces) |

The G2 parameters (two contexts, 128 physical vector registers, 8 barriers, 35-bit physical addresses, 32 sets × 8 ways) are unchanged from `image_g2_all.tcl`. Every port is connected to `HBM[0:31]` (`fpga/u280_multicore_hbm.cfg`), which striping and the single-outlet memory control rely on.

The compiler's defaults are not part of the image. Unless the proposed default change (`../default_change_tiled_pointwise.patch`, §8.1) is applied, programs on this image use the delivered lowering (`TNCA_TILED_POINTWISE=0`).

## Evidence before a link
| Check | Domain | Result | Where |
|---|---|---|---|
| XO packaging with this `image.tcl` (x86 host, Vivado 2023.2, tree `64e3c71`) | packaging | exit 0, no critical warnings; XO sha256 `a3020832…` | `package_dryrun.log` |
| Kernel synthesis of s4 (= this image without `HBM_STRIPED`) | screen | passes; 1,131,744 LUTs for the whole kernel | `../screens/s4_64e3c71d7f39/kernel_utilization.rpt` |
| Tile 0 placed and routed out of context in SLR0, 10 ns | screen | WNS +0.423 ns, 0 failing endpoints; 246,873 LUTs; 41,415 CLBs (75.35 % of SLR0) | `../screens/s4_64e3c71d7f39/` |
| Board-top RTL at the s4 parameters (two grids, memory controls 1/1, 1/32, 3/1, 3/32, coherent window and ownership handoff) | rtl | pass, after a bench defect was found and fixed (the shared-BO mirror sliced 5-bit AXI IDs as 3 bits) | `../geometry/` |
| Unit benches of each step | rtl | see G4_DESIGN.md §8.2 | `../step2/`, `../tag_tests/`, `../outlet_bench/` |
| Striped layout | rtl, unit | existing tests (`tools/r41/hbm_layout_tests.py`, `tb/fpga/tb_hbm_stripe.v`); the board-top bench cannot run it (it needs the 8 GiB window) | — |

Screens are out of context. Run 1's calibration from a tile screen to the full image is −0.9 to −1.0 ns for RAM-capture paths in a ≥ 96 % SLR (G4_DESIGN.md §4.2), so the +0.423 ns tile margin does not by itself predict closure at 100 MHz.

## How to link (the lead)
On the x86 host, per [`evidence/r41/stage2/x86_link/README.md`](../../../stage2/x86_link/README.md), from a clean tree at the commit that contains this file:

```sh
source ~/x86_link_env.sh
setsid nohup nice -n 5 bash flow/fpga/run_r40_attempt01.sh FRESH_BUILD_ROOT \
    evidence/r41/hbm_bw/g4/run2/image.tcl > /dev/null 2>&1 < /dev/null &
```

One link counts against the Stage 2 budget of five full runs. A new image is loaded onto the shared card only after the idle rule and the user's approval.
