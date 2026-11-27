#!/bin/bash
# img_chain.sh NAME SRC_TGZ IMAGE_TCL TILE... : kernel synthesis of an image (flow/fpga/synth_r40_multicore.tcl), then one
# out-of-context tile place and route at 10 ns per TILE index (flow/fpga/screen_r41_tile.tcl, tile t in SLR t).
# One lane: the caller runs at most two lanes (each Vivado job <= 4 threads set by the scripts).
set -u
name=$1 tgz=$2 image=$3; shift 3
source ~/x86_link_env.sh >/dev/null 2>&1
d=$HOME/r42img/$name
[ -e "$d" ] && { echo "exists $d"; exit 1; }
mkdir -p "$d/src" && tar xzf "$tgz" -C "$d/src" || exit 1
cd "$d"
start=$(date +%s)
nice -n 5 vivado -mode batch -nojournal -log synth.log -source src/flow/fpga/synth_r40_multicore.tcl -tclargs src "src/$image" out > /dev/null 2>&1
echo "$(date -u +%FT%TZ) $name synth exit $? $(( $(date +%s) - start )) s" >> ~/r42img/chain.log
for t in "$@"; do
    s2=$(date +%s)
    nice -n 5 vivado -mode batch -nojournal -log tile$t.log -source src/flow/fpga/screen_r41_tile.tcl -tclargs out/synth.dcp $t $t tile$t > /dev/null 2>&1
    echo "$(date -u +%FT%TZ) $name tile$t exit $? $(( $(date +%s) - s2 )) s" >> ~/r42img/chain.log
done
