# Full implementation on the x86 WSL host

The x86 host (Windows WSL2, Ubuntu 24.04, 16 cores, 31 GB RAM plus 32 GB swap since 2026-10-26) can run the whole image flow: packaging, `v++` link, synthesis, place and route. It has no card. Run 1 (the 100 MHz image) peaked at 31 GB on the FPGA host, so the host now has room for a link without relying on swap.

## What was installed
Vivado 2023.2 (67 GB), Vitis 2023.2 without the AI-engine tools, the `xilinx_u280_gen3x16_xdma_1_202211_1` platform and XRT were copied from the FPGA host into `$HOME/Xilinx` and `$HOME/opt/xilinx`. The flow scripts source `/tools/Xilinx/...` and `/opt/xilinx/...`, which do not exist here (those lines fail silently), so `x86_link_env.sh` puts the tools on `PATH` explicitly. Source it before the runner:

```sh
source ~/x86_link_env.sh
setsid nohup nice -n 5 bash flow/fpga/run_r40_attempt01.sh FRESH_BUILD_ROOT IMAGE_TCL > /dev/null 2>&1 < /dev/null &
```

The Ubuntu 24.04 gaps are closed in the user's home directory without root:
- `libtinfo.so.5` (Vivado) and `libboost_*.so.1.74.0` (`xclbinutil`), copied from the FPGA host into `~/Xilinx/compat-lib`;
- the `en_US.UTF-8` locale Vivado forces, in `~/.locale`;
- an empty device-data folder that Vivado expects under `SharedData`.

## What was verified (2026-10-26)
- `v++ v2023.2`, `vivado v2023.2` and `xclbinutil` (XRT 2.16.204) start; the U280 platform is found.
- The xcu280 part is available without a license.
- `package_xo.tcl` on the run 1 source (the archive hash `4a2dcaf6…` matches run 1) with the run 1 image parameters packages a kernel in about 20 s with no critical warnings. Its XO file hash differs from run 1's (an XO is a zip with timestamps), so reproducibility is judged on the link result, not on that hash.
- Not verified here: a complete `v++` link. It has not been run on this host.

## Cautions
- Use at most one link at a time; keep `VPP_JOBS=2`.
- A link counts against the Stage 2 budget of five full runs (one used).
- A new image is loaded onto the shared card only after the idle rule and the user's approval.
