# Environment for Vivado/Vitis 2023.2 and the U280 platform on the x86 WSL host.
# The installs were copied from the FPGA host and live under $HOME, while the flow
# scripts source /tools/Xilinx/... (missing here: those sources fail silently), so
# the tools are put on PATH explicitly. Source this before flow/fpga/run_r40_attempt01.sh.
export LD_LIBRARY_PATH=$HOME/Xilinx/compat-lib:${LD_LIBRARY_PATH:-}   # libtinfo.so.5 (Ubuntu 24.04 lacks it)
export LANG=en_US.UTF-8 LOCPATH=$HOME/.locale                          # the locale Vivado forces
export XILINX_VIVADO=$HOME/Xilinx/Vivado/2023.2 XILINX_VITIS=$HOME/Xilinx/Vitis/2023.2
export XILINX_XRT=$HOME/opt/xilinx/xrt XILINX_PLATFORMS=$HOME/opt/xilinx/platforms PLATFORM_REPO_PATHS=$HOME/opt/xilinx/platforms
export PATH=$XILINX_VITIS/bin:$XILINX_VIVADO/bin:$XILINX_XRT/bin:$PATH
export TNCA_BOARD=u280
