# r41s2 base (544aed8, C1 applied): kernel synthesis and tile-1 screen

Both runs were on the FPGA host's CPU through `launches/vivado_slot.sh`. Neither touched the card. The raw reports are in `base-544aed8-evidence.tgz`; the checkpoints stay on the host under `~/r41s2-base-544aed841a95`.

| Step | Run (UTC) | Result |
|---|---|---|
| Kernel synthesis, `synth_r40_multicore.tcl` with the attempt-3 image parameters | 2026-10-21 14:19–14:48 | Unplaced WNS +1.740 ns, TNS 0, at 10 ns. This is the same as the d2cc850 kernel synthesis |
| Tile 1 OOC place and route, `screen_r41_tile.tcl` in SLR1 at 10 ns | 2026-10-21 14:52–16:33 | Placed WNS +0.530 ns. **Routed WNS +0.015 ns, TNS 0, WHS +0.019 ns, 0 failing endpoints.** 249,646 LUTs, 68,399 registers, 39.5 BRAM tiles |

The worst routed paths all start at `u_core/x_pm_idx_reg` and end at the writeback data registers `w_sdata` / `w_vdata`: 18–24 logic levels, about 76 % route delay. That is the F04 writeback family. The ten worst paths have between 0.015 and 0.154 ns of slack.

What this does not show:
- **It is optimistic.** The screen places one tile alone in a whole SLR. There is no shell, no other tile, no SLR crossing and none of SLR0's static-region density (95.24 % CLB packing in A3). The routed attempt-3 image failed 100 MHz with WNS −1.002 ns, and most of its residual census was inside the tiles.
- **It is not calibrated.** No pre-C1 tile was screened the same way. Without that baseline, a pass here cannot be credited to C1.
- **It is not a 100 MHz result.** F01–F06 and F08–F10 are not implemented on this branch. The Stage 2 line has been paused since this run; no further jobs were launched.
