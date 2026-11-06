#!/bin/bash
# F07 (C1, 658c811) unit OOC screen: tnca_nuca_slice at the B40 image's slice
# parameters, current RTL vs C1, synthesis-only at 10 ns. Never touches the card.
set -u
D=$HOME/r41s1_ooc_f07
G="SETS=32 SET_W=5 WAYS=8 WAY_W=3 MSHR_N=8 MSHR_W=3 TARGET_N=16 TARGET_W=4 TAG_W=5 SLICE_ID=0 FULL_LINE_TAG=1 MAINTENANCE=1 REGISTER_VICTIM_OWNER=1 REGISTER_VICTIM_SELECT=1 CACHE_DATA_PIPELINE=1 REGISTER_VICTIM_RANK=1 DCO_REGISTERED_RANGES=1"
source /tools/Xilinx/Vivado/2023.2/settings64.sh >/dev/null 2>&1
for side in base c1; do
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 16 ]; do sleep 60; done
  root=$D/$side; out=$D/out/$side; mkdir -p $out
  files=$(cd $root/rtl && ls common/*.v core/*.v gen/*.v nuca/*.v noc/*.v tile/*.v fpga/*.v *.v)
  echo "$(date -u +%FT%TZ) start $side avail=$(free -g | awk '/^Mem:/{print $7}')G" >> $D/run.log
  OOC_SV=1 OOC_GENERICS="$G" nice -n 10 vivado -mode batch -nojournal -nolog \
    -source $root/flow/synth/vivado_ooc_timing.tcl \
    -tclargs tnca_nuca_slice 10 xcu280-fsvh2892-2L-e $out $root/rtl $files > $out/vivado.log 2>&1
  echo "$(date -u +%FT%TZ) end $side exit $?" >> $D/run.log
done
echo F07-DONE >> $D/run.log
