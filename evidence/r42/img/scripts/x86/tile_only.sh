#!/bin/bash
# tile_only.sh NAME: out-of-context tile 0 place and route from an existing kernel synthesis ~/r42img/NAME/out/synth.dcp
source ~/x86_link_env.sh >/dev/null 2>&1
cd $HOME/r42img/$1 || exit 1
s2=$(date +%s)
nice -n 5 vivado -mode batch -nojournal -log tile0.log -source src/flow/fpga/screen_r41_tile.tcl -tclargs out/synth.dcp 0 0 tile0 > /dev/null 2>&1
echo "$(date -u +%FT%TZ) $1 tile0 exit $? $(( $(date +%s) - s2 )) s" >> ~/r42img/chain.log
