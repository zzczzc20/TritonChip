#!/bin/bash
# gates_top.sh TREE OUTDIR  (GB10, nice 19) -- tb_tnca_top (2x2 system: cores, slices, TMU, mesh) with the programs of `make top` at
# CLKFIX2 = 0 (reference) and with the image-level bits: 1024 (bit 10, TMU input registers), 2048 (bit 11, fast router), 3072 (both),
# each with DCO_REGISTER_WRITES 0 and 1 (DCO_TLW_PIPE_STAGES 2 when 1). Reports the cycle counts of every run.
R=$1; O=$2; mkdir -p $O; cd $R || exit 2
source ~/tc_env/env.sh >/dev/null 2>&1
RTL=$R/rtl
RTLY="-y $RTL/common -y $RTL/core -y $RTL/gen -y $RTL/nuca -y $RTL/noc -y $RTL/tile -y $RTL/fpga -y $RTL -I $RTL/noc"
make BUILD=$O/b t10_mesh.hex t24_amo_mesh.hex >/dev/null 2>&1; ls $O/b/*.hex 2>/dev/null | head -3
for rw in 0 1; do for m in 0 1024 2048 3072; do
  tlwp=0; [ $rw = 1 ] && tlwp=2
  v=$O/top_rw${rw}_m${m}.vvp
  iverilog -g2005 -I $RTL/common $RTLY -P tb_tnca_top.CLKFIX2=$m -P tb_tnca_top.DCO_RW=$rw -P tb_tnca_top.DCO_TLWP=$tlwp -o $v tb/top/tb_tnca_top.v $RTL/tnca_top.v tb/common/tnca_dram_model.v 2>&1 | head -5
  for p in t10_mesh t24_amo_mesh; do
    extra="+needmesh"; [ $p = t10_mesh ] && extra="+needevict +needmesh"
    for roles in "" "+allroles"; do
      echo "rw=$rw CLKFIX2=$m $p $roles: $(nice -n 19 vvp $v +hex=$O/b/$p.hex +maxcycles=2000000 $roles $extra 2>&1 | grep -E '^(PASS|FAIL|cycles=|tb_)' | tr '\n' ' ')"
    done
  done
done; done
