#!/bin/bash
# gates_noc.sh TREE_ROOT OUTDIR  (GB10, run under nice 19) -- tnca_router_fast / TMU input register gates:
#   router miter (random traffic, both routers), tb_mesh at LINK_STAGES 0 and 2 with ROUTER_FAST 0/1 (3 seeds),
#   tb_tmu_inreg, tb_top (t10_mesh, t24_amo_mesh) at CLKFIX2 = 0, 2048 (TMU input regs), 1024... see below.
R=$1; O=$2; mkdir -p $O; cd $R || exit 2
source ~/tc_env/env.sh >/dev/null 2>&1
RTL=$R/rtl
RTLY="-y $RTL/common -y $RTL/core -y $RTL/gen -y $RTL/nuca -y $RTL/noc -y $RTL/tile -y $RTL/fpga -y $RTL -I $RTL/noc -I $RTL/common"
IV="iverilog -g2012 -I $RTL/common"
echo "== router miter"; $IV -P tb_router_miter.CYCLES=30000 -o $O/miter.vvp tb/clk2/tb_router_miter.v $RTL/noc/tnca_router.v $RTL/noc/tnca_router_fast.v $RTL/common/tnca_fifo.v && nice -n 19 vvp $O/miter.vvp | tail -3
echo "== tmu inreg"; $IV -I $RTL/nuca -o $O/tmu.vvp tb/clk2/tb_tmu_inreg.v $RTL/nuca/tnca_tmu.v && nice -n 19 vvp $O/tmu.vvp | tail -6
for ls in 0 1 2; do for rf in 0 1; do
  echo "== tb_mesh LINK_STAGES=$ls ROUTER_FAST=$rf"
  iverilog -g2005 -I $RTL/common $RTLY -P tb_mesh.LINK_STAGES=$ls -P tb_mesh.ROUTER_FAST=$rf -o $O/mesh_${ls}_${rf}.vvp tb/noc/tb_mesh.v $RTL/noc/tnca_mesh.v
  for sd in 1 2 3; do nice -n 19 vvp $O/mesh_${ls}_${rf}.vvp +npkt=2000 +seed=$sd | tail -2; done
done; done
